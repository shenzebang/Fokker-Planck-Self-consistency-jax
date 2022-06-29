import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import flax.optim
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import jax.random as random
from flax import serialization
import flax.linen as nn
import jax
from jax import jvp, grad, value_and_grad, vjp
from jax.experimental.ode import odeint

from plot_utils import *
from model.neural_ode_model_flax import DenseNet3
from core import distribution, potential
from tqdm import tqdm, trange
from test_utils import eval_Gaussian_score_and_log_density_FP
import pandas as pd
from config import args_parser
import json



def train_PINN(args, net: nn.Module, init_distribution: distribution.Distribution, target_potential: potential.Potential, test_data, key):
    key1, key2, key3 = random.split(key, 3)

    # randomly initialize the model, autobatching is included here
    params = net.init(key1, jnp.zeros(1), init_distribution.sample(1, key2))

    params_flat, params_tree = tree_flatten(params)

    def gv(params, data):
        f = lambda _x, _t, _params: net.apply(_params, _t, _x) # f is log \rho
        f_t = grad(f, argnums=1)
        v_f_t = jax.vmap(f_t, in_axes=[0, None, None])
        f_x = grad(f, argnums=0)
        f_x_grad_V_p_f_x = lambda _x, _t, _params: jnp.dot(f_x(_x, _t, _params), target_potential.gradient(_x) + args.diffusion_coefficient * f_x(_x, _t, _params))
        v_f_x_grad_V_p_f_x = jax.vmap(f_x_grad_V_p_f_x, in_axes=[0, None, None])

        grad_0 = [jnp.zeros_like(_var) for _var in params_flat]
        loss_0 = jnp.zeros(1)

        states_0 = [grad_0, loss_0]
        def ode_func(states, t):
            laplacian_V = divergence_fn(target_potential.gradient, data)
            def laplacian_f(_params):
                f_x_t_params = lambda _x: f_x(_x, t, _params)
                return divergence_fn(f_x_t_params, data)

            def loss_t(_params):
                v1 = v_f_t(data, t, _params)
                v2 = - v_f_x_grad_V_p_f_x(data, t, _params)
                v3 = - args.diffusion_coefficient * laplacian_f(_params)
                v4 = - laplacian_V
                v = v1 + v2 + v3 + v4
                return jnp.mean(v ** 2)
            grad_t = grad(loss_t, argnums=0)

            return [grad_t(params), loss_t(params)]

        tspace = jnp.array((0., args.total_evolving_time))
        result_forward = odeint(ode_func, states_0, tspace, atol=args.ODE_tolerance, rtol=args.ODE_tolerance)
        # grad_T = result_forward[0][1]
        grad_T = tree_unflatten(params_tree, [_var[1] for _var in result_forward[0]])
        loss_T = result_forward[1][1]
        return grad_T, loss_T

    opt_def = flax.optim.Adam(learning_rate=args.learning_rate)
    opt = opt_def.create(params)
    gv = jax.jit(gv)

    # define train_op
    def train_op(_opt, x):

        g, v = gv(_opt.target, x)

        return v, _opt.apply_gradient(g)

    # train_op = jax.jit(train_op)

    # training process
    running_avg_loss = 0.

    score_testing_error_list = []
    log_density_testing_error_list = []
    step_list = []
    loss_list = []
    save_dir = f"./save/COLT"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = f"./save/COLT/PINN.csv"

    # unpack the test data
    time_stamps, grid_points, gaussian_scores_on_grid, gaussian_log_density_on_grid = test_data
    for step in trange(args.number_of_iterations):
        key, subkey = random.split(key)
        data = jax.random.uniform(subkey, (args.train_batch_size, init_distribution.dim), minval=-10, maxval=10)
        # data = init_distribution.sample(batch_size)
        # idx = jax.random.choice(subkey, grid_points.shape[0], (batch_size,))

        loss, opt = train_op(opt, data)
        running_avg_loss += loss[0]
        if step % args.test_frequency == args.test_frequency - 1:
            f_x = grad(net.apply, argnums=2)
            v_f_x = jax.vmap(f_x, in_axes=[None, None, 0])
            vv_f_x = jax.vmap(v_f_x, in_axes=[None, 0, None])
            scores_pred = vv_f_x(opt.target, time_stamps, grid_points)
            score_testing_error = jnp.mean(jnp.sum((scores_pred - gaussian_scores_on_grid) ** 2, axis=(2,)))

            v_f = jax.vmap(net.apply, in_axes=[None, 0, None])
            log_density_pred = v_f(opt.target, time_stamps, grid_points)
            # log_density_testing_error = jnp.mean((log_density_pred - gaussian_log_density_on_grid) ** 2)

            log_density_testing_error = jnp.mean(
                jnp.abs(jnp.exp(gaussian_log_density_on_grid[1:]) - jnp.exp(log_density_pred[1:]))
            )  # ignore time 0

            # testing_error = []
            # for i, t in enumerate(time_stamps):
            #     score_pred = v_f_x(opt.target, t, grid_points)
            #     loss_t = jnp.mean(jnp.sum((score_pred - gaussian_scores_on_grid[i]) ** 2, axis=(1,)))
            #     testing_error.append(loss_t)
            #     print(t)
            # testing_error = jnp.mean(jnp.array(testing_error))

            print('Step %04d  Loss %.5f  Score Error %.5f Log-density Error %.5f' %     (step + 1,
                                                            running_avg_loss    /   (step + 1),
                                                            score_testing_error,
                                                            log_density_testing_error
                                                            )
                  )
            step_list.append(step + 1)
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
    net = DenseNet3(1, key1) # For PINN, we parameterize the log density

    ############### testing data ###############
    # compute the testing data
    test_time_stamps = jnp.linspace(0, args.total_evolving_time, num=11)
    x, y = jnp.linspace(-args.test_domain_size, args.test_domain_size, num=201), jnp.linspace(-args.test_domain_size,
                                                                                              args.test_domain_size,
                                                                                              num=201)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([jnp.reshape(xx, (-1)), jnp.reshape(yy, (-1))], axis=1)
    gaussian_scores_on_grid, gaussian_log_density_on_grid \
        = eval_Gaussian_score_and_log_density_FP(sigma0, mu0, sigma_target, mu_target, test_time_stamps, grid_points,
                                                 args.diffusion_coefficient)
    test_data = [test_time_stamps, grid_points, gaussian_scores_on_grid, gaussian_log_density_on_grid]

    ############### training ###############
    params = train_PINN(args, net, init_distribution, target_potential, test_data, key2)

    ############### testing & logging ###############
    # This part is included in the training section