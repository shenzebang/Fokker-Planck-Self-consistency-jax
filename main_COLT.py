import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import flax.optim
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import jax.random as random
from flax import serialization

import jax
from jax import jvp, grad, value_and_grad, vjp
from jax.experimental.ode import odeint

from utils import *
from model.neural_ode_model_flax import DenseNet2
from core import distribution, potential
from tqdm import trange
from test_utils import eval_Gaussian_score_and_log_density_FP, log_density
import pandas as pd
import json
from config import args_parser
from plot_utils import plot_velocity_field_2d, plot_density_contour_2d


def train_nwgf(args, net, init_distribution: distribution.Distribution, target_potential: potential.Potential, test_data, key: jnp.ndarray):

    key1, key2, key3 = jax.random.split(key, 3)

    # randomly initialize the model, autobatching is included here
    params = net.init(key1, jnp.zeros(1), init_distribution.sample(1, key2))

    params_flat, params_tree = tree_flatten(params)

    def nwgf_gv(params, data):
        bar_f = lambda _x, _t, _params: net.apply(_params, _t, _x) - target_potential.gradient(_x)
        f = lambda _x, _t, _params: net.apply(_params, _t, _x)
        # compute x(T) by solve IVP (I) & compute the actor loss
        # ================ Forward ===================
        x_0 = data
        xi_0 = init_distribution.score(x_0)
        loss_0 = jnp.zeros(1)
        states_0 = [x_0, xi_0, loss_0]

        def ode_func1(states, t):
            x = states[0]
            xi = states[1]
            f_t_theta = lambda _x: f(_x, t, params)
            bar_f_t_theta = lambda _x: bar_f(_x, t, params)
            dx = bar_f_t_theta(x)

            def h_t_theta(in_1, in_2):
                # in_1 is xi
                # in_2 is x
                div_bar_f_t_theta = lambda _x: divergence_fn(bar_f_t_theta, _x).sum(axis=0)
                grad_div_fn = grad(div_bar_f_t_theta)
                h1 = - grad_div_fn(in_2)
                _, vjp_fn = vjp(bar_f_t_theta, in_2)
                h2 = - vjp_fn(in_1)[0]
                return h1 + h2

            dxi = h_t_theta(xi, x)

            def g_t(in_1, in_2):
                # in_1 is xi
                # in_2 is x
                f_t_theta_in_2 = f_t_theta(in_2)
                reg = args.reg_f*jnp.sum(f_t_theta_in_2 ** 2, axis=(1,))
                return jnp.mean(
                    jnp.sum((f_t_theta_in_2 + args.diffusion_coefficient * in_1) ** 2, axis=(1, )) + reg
                )

            dloss = g_t(xi, x)

            return [dx, dxi, dloss]

        tspace = jnp.array((0., args.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=args.ODE_tolerance, rtol=args.ODE_tolerance)
        x_T = result_forward[0][1]
        xi_T = result_forward[1][1]
        loss_f = result_forward[2][1]
        # ================ Forward ===================

        # ================ Backward ==================
        # compute dl/d theta via adjoint method
        a_T = jnp.zeros_like(x_T)
        b_T = jnp.zeros_like(x_T)
        grad_T = [jnp.zeros_like(_var) for _var in params_flat]
        loss_T = jnp.zeros(1)
        states_T = [x_T, a_T, b_T, xi_T, loss_T, grad_T]

        def ode_func2(states, t):
            t = args.total_evolving_time - t
            x = states[0]
            a = states[1]
            b = states[2]
            xi = states[3]

            f_t = lambda _x, _params: f(_x, t, _params)
            bar_f_t = lambda _x, _params: bar_f(_x, t, _params)
            dx = bar_f_t(x, params)

            _, vjp_fx_fn = vjp(lambda _x: bar_f_t(_x, params), x)
            vjp_fx_a = vjp_fx_fn(a)[0]
            _, vjp_ftheta_fn = vjp(lambda _params: bar_f_t(x, _params), params)
            vjp_ftheta_a = vjp_ftheta_fn(a)[0]

            def h_t(in_1, in_2, in_3):
                # in_1 is xi
                # in_2 is x
                # in_3 is theta
                bar_f_t_theta = lambda _x: bar_f_t(_x, in_3)
                div_bar_f_t_theta = lambda _x: divergence_fn(bar_f_t_theta, _x).sum(axis=0)
                grad_div_fn = grad(div_bar_f_t_theta)
                h1 = - grad_div_fn(in_2)
                _, vjp_fn = vjp(bar_f_t_theta, in_2)
                h2 = - vjp_fn(in_1)[0]
                return h1 + h2

            _, vjp_hxi_fn = vjp(lambda _xi: h_t(_xi, x, params), xi)
            vjp_hxi_b = vjp_hxi_fn(b)[0]
            _, vjp_hx_fn = vjp(lambda _x: h_t(xi, _x, params), x)
            vjp_hx_b = vjp_hx_fn(b)[0]
            _, vjp_htheta_fn = vjp(lambda _params: h_t(xi, x, _params), params)
            vjp_htheta_b = vjp_htheta_fn(b)[0]

            def g_t(in_1, in_2, in_3):
                # in_1 is xi
                # in_2 is x
                # in_3 is theta
                f_t_in_2_in_3 = f_t(in_2, in_3)
                reg = args.reg_f * jnp.sum(f_t_in_2_in_3 ** 2, axis=(1,))
                return jnp.mean(
                    jnp.sum((f_t_in_2_in_3 + args.diffusion_coefficient * in_1) ** 2, axis=(1, )) + reg
                )

            dxig = grad(g_t, argnums=0)
            dxg = grad(g_t, argnums=1)
            dthetag = grad(g_t, argnums=2)

            da = - vjp_fx_a - vjp_hx_b - dxg(xi, x, params)
            db = - vjp_hxi_b - dxig(xi, x, params)
            dxi = h_t(xi, x, params)
            dloss = g_t(xi, x, params)[None]

            vjp_ftheta_a_flat, _ = tree_flatten(vjp_ftheta_a)
            vjp_htheta_b_flat, _ = tree_flatten(vjp_htheta_b)
            dthetag_flat, _ = tree_flatten(dthetag(xi, x, params))
            # print(len(vjp_ftheta_a_flat), len(vjp_htheta_b_flat), len(dthetag_flat))
            dgrad = [_dgrad1/x.shape[0] + _dgrad2/x.shape[0] + _dgrad3 for _dgrad1, _dgrad2, _dgrad3 in
                     zip(vjp_ftheta_a_flat, vjp_htheta_b_flat, dthetag_flat)]
            # dgrad = vjp_ftheta_a + vjp_htheta_b + dthetag(xi, x, params)

            return [-dx, -da, -db, -dxi, dloss, dgrad]

        # ================ Backward ==================
        tspace = jnp.array((0., args.total_evolving_time))
        result_backward = odeint(ode_func2, states_T, tspace, atol=args.ODE_tolerance, rtol=args.ODE_tolerance)

        grad_T = tree_unflatten(params_tree, [_var[1] for _var in result_backward[5]])
        x_0_b = result_backward[0][1]
        xi_0_b = result_backward[3][1]
        error_x = jnp.mean(jnp.sum((x_0_b - x_0).reshape(x_0.shape[0], -1) ** 2, axis=(1,)))
        error_xi = jnp.mean(jnp.sum((xi_0 - xi_0_b).reshape(xi_0.shape[0], -1) ** 2, axis=(1,)))

        loss_b = result_backward[4][1]
        return grad_T, loss_b, loss_f, error_x, error_xi

    opt_def = flax.optim.Adam(learning_rate=args.learning_rate)
    opt = opt_def.create(params)
    nwgf_gv = jax.jit(nwgf_gv)

    # define train_op
    def train_op(_opt, x):

        g, v_b, v_f, error_x, error_xi = nwgf_gv(_opt.target, x)

        return v_b, _opt.apply_gradient(g), v_f, error_x, error_xi

    # train_op = jax.jit(train_op)

    # training process
    running_avg_loss = 0.
    running_error_xi = 0.
    running_error_x = 0.

    # unpack the test data
    time_stamps, grid_points, gaussian_scores_on_grid, gaussian_log_density_on_grid = test_data

    def test_op(params, Gaussian_score, gaussian_log_density_on_grid):

        v_net_apply = jax.vmap(net.apply, in_axes=[None, 0, None])
        negative_scores_pred = v_net_apply(params, time_stamps, grid_points)

        velocity = lambda  _params, _x, _t,: net.apply(_params, _t, _x) - target_potential.gradient(_x)

        v_log_density = jax.vmap(init_distribution.logdensity)

        log_density_pred = log_density(params, velocity, v_log_density, time_stamps, grid_points)


        score_error = jnp.mean(jnp.sum((negative_scores_pred + Gaussian_score) ** 2, axis=(2,)))
        log_density_error = jnp.mean(jnp.sum(jnp.abs(jnp.exp(gaussian_log_density_on_grid[1:]) - jnp.exp(log_density_pred[1:])), axis=1))# ignore time 0
        return score_error, log_density_error


    def generate_density_data(data, params, end_T, n_frames=100):
        bar_f = lambda _x, _t: net.apply(params, _t, _x) - target_potential.gradient(_x)

        states_0 = [data]

        def ode_func1(states, t):
            x = states[0]
            dx = bar_f(x, t)
            return [dx]

        tspace = jnp.linspace(0, end_T, n_frames)

        result_forward = odeint(ode_func1, states_0, tspace, atol=1e-3, rtol=1e-3)

        return result_forward

    j_generate_density_data = jax.jit(generate_density_data, static_argnames=['end_T', 'n_frames'])
    score_testing_error_list = []
    log_density_testing_error_list = []
    loss_list = []
    step_list = []
    save_dir = f"./save/COLT"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = f"./save/COLT/NWGF.csv"
    keys = jax.random.split(key3, args.number_of_iterations)
    for step in trange(args.number_of_iterations):
        _key1, _key2 = jax.random.split(keys[step], 2)
        data = init_distribution.sample(args.train_batch_size, _key1)
        loss_b, opt, loss_f, error_x, error_xi = train_op(opt, data)
        running_avg_loss += loss_b[0]
        running_error_xi += error_xi
        running_error_x += error_x

        if step % args.plot_frequency == args.plot_frequency - 1:
            # plot velocity field
            f_velocity = lambda _x, _t,: net.apply(params, _t, _x) - target_potential.gradient(_x)
            plot_velocity_field_2d(args, f_velocity)

            # plot density contour
            init_data = init_distribution.sample(1000, _key2)
            density_data = j_generate_density_data(init_data, opt.target, end_T=args.total_evolving_time)
            plot_density_contour_2d(args, density_data)

        if step % args.test_frequency == args.test_frequency - 1:
            score_testing_error, log_density_testing_error \
                = test_op(opt.target, gaussian_scores_on_grid, gaussian_log_density_on_grid)

            print('Step %04d  Loss %.5f Error_xi %.5f Error_x %.5f Score Error %.5f Log-density Error %.5f' %     (step + 1,
                                                            running_avg_loss    /   (step + 1),
                                                            running_error_xi    /   (step + 1),
                                                            running_error_x     /   (step + 1),
                                                            score_testing_error,
                                                            log_density_testing_error
                                                            )
                  )
            step_list.append(step+1)
            score_testing_error_list.append(score_testing_error)
            log_density_testing_error_list.append(log_density_testing_error)
            loss_list.append(running_avg_loss/(step + 1))

        if step % args.save_frequency == args.save_frequency - 1:
            steps = jnp.array(step_list)
            score_accs = jnp.array(score_testing_error_list)
            log_density_accs = jnp.array(log_density_testing_error_list)
            losses = jnp.array(loss_list)
            result = pd.DataFrame(jnp.stack([steps, score_accs, losses, log_density_accs], axis=1),
                                columns=['steps', 'score_accs', 'losses', 'log-density accs'])
            result.to_csv(save_file, index=False)



    steps = jnp.array(step_list)
    score_accs = jnp.array(score_testing_error_list)
    log_density_accs = jnp.array(log_density_testing_error_list)
    losses = jnp.array(loss_list)
    result = pd.DataFrame(jnp.stack([steps, score_accs, losses, log_density_accs], axis=1),
                          columns=['steps', 'score_accs', 'losses', 'log-density accs'])
    result.to_csv(save_file, index=False)

    return opt.target






if __name__ == '__main__':
    args = args_parser()

    save_directory = f"./{args.plot_save_directory}/{args.PDE}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    with open(save_directory + '/config.json', 'w') as f:
        json.dump(vars(args), f)

    key = random.PRNGKey(args.seed)
    key1, key2 = random.split(key)

    ############### initial distribution ###############
    mu0 = jnp.array([-4.0, -4.0])
    sigma0 = jnp.diag(jnp.array([0.7, 1.3]))
    init_distribution = distribution.Gaussian(mu0, sigma0)

    ############### drifting term ###############
    mu_target = jnp.array([4.0, 4.0])
    sigma_target = jnp.diag(jnp.array([1.1, 0.9]))
    target_potential = potential.QuadraticPotential(mu_target, sigma_target)

    ############### model ###############
    net = DenseNet2(init_distribution.dim, key1)

    ############### testing data ###############
    test_time_stamps = jnp.linspace(0, args.total_evolving_time, num=11)
    x, y = jnp.linspace(-args.test_domain_size, args.test_domain_size, num=201), jnp.linspace(-args.test_domain_size, args.test_domain_size, num=201)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([jnp.reshape(xx, (-1)), jnp.reshape(yy, (-1))], axis=1)
    gaussian_scores_on_grid, gaussian_log_density_on_grid \
        = eval_Gaussian_score_and_log_density_FP(sigma0, mu0, sigma_target, mu_target, test_time_stamps, grid_points, args.diffusion_coefficient)
    test_data = [test_time_stamps, grid_points, gaussian_scores_on_grid, gaussian_log_density_on_grid]

    ############### training ###############
    params = train_nwgf(args, net, init_distribution, target_potential, test_data, key2)

    ############### testing & logging ###############
    # This part is included in the training section

