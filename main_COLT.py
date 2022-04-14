import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import flax.optim
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import jax.random as random
from flax import serialization

import jax
from jax import jvp, grad, value_and_grad, vjp
from jax.experimental.ode import odeint

from utils import *
from plot_utils import *
from model.neural_ode_model_flax import UNet, DenseNet, G2GNet, DenseNet2
from core import distribution, potential
# import torchvision.transforms as transforms
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# from torchvision.datasets import MNIST
# from sampler import ode_sampler
# from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from test_utils import eval_Gaussian_score_FP, test_Gaussian_score

import functools


# global configuration

num_iterations = 5000
batch_size = 64
lr = 2e-3
tolerance = 1e-5
T = 3.
discount = 1.
scale = 1.
reg_f = 0.
testing_freq = 10




def train_nwgf(net, init_distribution: distribution.Distribution, target_potential: potential.Potential, test_data):

    # randomly initialize the model, autobatching is included here
    params = net.init(subkey, jnp.zeros(1), init_distribution.sample(1))

    params_flat, params_tree = tree_flatten(params)

    def nwgf_gv(params, data):
        bar_f = lambda _x, _t, _params: net.apply(_params, _t, _x) * scale - target_potential.gradient(_x)
        f = lambda _x, _t, _params: net.apply(_params, _t, _x) * scale
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
                return jnp.mean(jnp.sum((f_t_theta_in_2 + in_1) ** 2, axis=(1, )) + reg_f*jnp.sum(f_t_theta_in_2 ** 2, axis=(1,))) * (discount ** t)

            dloss = g_t(xi, x)

            return [dx, dxi, dloss]

        tspace = jnp.array((0., T))
        result_forward = odeint(ode_func1, states_0, tspace, atol=tolerance, rtol=tolerance)
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
            t = T - t
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
                return jnp.mean(jnp.sum((f_t_in_2_in_3 + in_1) ** 2, axis=(1, )) + reg_f*jnp.sum(f_t_in_2_in_3 ** 2, axis=(1, ))) * (discount ** t)

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
        tspace = jnp.array((0., T))
        result_backward = odeint(ode_func2, states_T, tspace, atol=tolerance, rtol=tolerance)

        grad_T = tree_unflatten(params_tree, [_var[1] for _var in result_backward[5]])
        x_0_b = result_backward[0][1]
        xi_0_b = result_backward[3][1]
        error_x = jnp.mean(jnp.sum((x_0_b - x_0).reshape(x_0.shape[0], -1) ** 2, axis=(1,)))
        error_xi = jnp.mean(jnp.sum((xi_0 - xi_0_b).reshape(xi_0.shape[0], -1) ** 2, axis=(1,)))

        loss_b = result_backward[4][1]
        return grad_T, loss_b, loss_f, error_x, error_xi

    opt_def = flax.optim.Adam(learning_rate=lr)
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
    time_stamps, grid_points, gaussian_scores_on_grid = test_data

    def test_op(params, time_stamps, grid_points, Gaussian_score):
        v_net_apply = jax.vmap(net.apply, in_axes=[None, 0, None])
        negative_scores_pred = v_net_apply(params, time_stamps, grid_points)

        return jnp.mean(jnp.sum((negative_scores_pred + Gaussian_score) ** 2, axis=(2,)))

    test_op = jax.jit(test_op)

    for step in trange(num_iterations):
        data = init_distribution.sample(batch_size)
        loss_b, opt, loss_f, error_x, error_xi = train_op(opt, data)
        running_avg_loss += loss_b[0]
        running_error_xi += error_xi
        running_error_x += error_x
        if step % testing_freq == testing_freq - 1:
            testing_error = test_op(opt.target, time_stamps, grid_points, gaussian_scores_on_grid)

            print('Step %04d  Loss %.5f Error_xi %.5f Error_x %.5f Test %.5f' %     (step + 1,
                                                            running_avg_loss    /   (step + 1),
                                                            running_error_xi    /   (step + 1),
                                                            running_error_x     /   (step + 1),
                                                            testing_error
                                                            )
                  )

    return opt.target






if __name__ == '__main__':
    key = random.PRNGKey(1)
    key, subkey = random.split(key)

    dim = 2
    mu0 = jnp.zeros((dim,))
    sigma0 = jnp.eye(dim)
    init_distribution = distribution.Gaussian(mu0, sigma0, subkey)

    key, subkey = random.split(key)
    mu_target = jnp.zeros((dim,)) + 4
    # sigma_target = jax.random.normal(subkey, (dim, dim))
    # sigma_target = jnp.eye(dim)
    sigma_target = jnp.diag(jnp.array([1.1, 0.9]))
    target_potential = potential.QuadraticPotential(mu_target, sigma_target)

    # construct the NODE
    net = DenseNet2(init_distribution.dim, key)

    # compute the testing data
    test_time_stamps = jnp.linspace(0, T, num=101)
    x, y = jnp.linspace(-10, 10, num=101), jnp.linspace(-10, 10, num=101)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([jnp.reshape(xx, (-1)), jnp.reshape(yy, (-1))], axis=1)
    gaussian_scores_on_grid = eval_Gaussian_score_FP(sigma0, mu0, sigma_target, mu_target, test_time_stamps, grid_points, 1)
    test_data = [test_time_stamps, grid_points, gaussian_scores_on_grid]
    params = train_nwgf(net, init_distribution, target_potential, test_data)

    # plot_result(net, params, init_distribution, target_potential, T)
