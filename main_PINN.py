import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from model.neural_ode_model_flax import UNet, DenseNet, G2GNet, DenseNet3
from core import distribution, potential
# import torchvision.transforms as transforms
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# from torchvision.datasets import MNIST
# from sampler import ode_sampler
# from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from test_utils import eval_Gaussian_score_FP, test_Gaussian_score
dim = 2
import functools


# global configuration

num_iterations = 5000
batch_size = 64
lr = 1e-2
tolerance = 1e-5
T = 3.
discount = 1.
scale = 1.
reg_f = 1
testing_freq = 10

# we use f (net with params) to parameterize log alpha



def train_PINN(net, init_distribution: distribution.Distribution, target_potential: potential.Potential, test_data, key):
    key, subkey = random.split(key)

    # randomly initialize the model, autobatching is included here
    params = net.init(subkey, jnp.zeros(1), init_distribution.sample(1))

    params_flat, params_tree = tree_flatten(params)

    def gv(params, data):
        f = lambda _x, _t, _params: net.apply(_params, _t, _x)
        f_t = grad(f, argnums=1)
        v_f_t = jax.vmap(f_t, in_axes=[0, None, None])
        f_x = grad(f, argnums=0)
        f_x_grad_V_p_f_x = lambda _x, _t, _params: jnp.dot(f_x(_x, _t, _params), target_potential.gradient(_x) + f_x(_x, _t, _params))
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
                v3 = - laplacian_f(_params)
                v4 = - laplacian_V
                v = v1 + v2 + v3 + v4
                return jnp.mean(v ** 2)
            grad_t = grad(loss_t, argnums=0)

            return [grad_t(params), loss_t(params)]

        tspace = jnp.array((0., T))
        result_forward = odeint(ode_func, states_0, tspace, atol=tolerance, rtol=tolerance)
        # grad_T = result_forward[0][1]
        grad_T = tree_unflatten(params_tree, [_var[1] for _var in result_forward[0]])
        loss_T = result_forward[1][1]
        return grad_T, loss_T

    opt_def = flax.optim.Adam(learning_rate=lr)
    opt = opt_def.create(params)
    gv = jax.jit(gv)

    # define train_op
    def train_op(_opt, x):

        g, v = gv(_opt.target, x)

        return v, _opt.apply_gradient(g)

    # train_op = jax.jit(train_op)

    # training process
    running_avg_loss = 0.

    # unpack the test data
    time_stamps, grid_points, gaussian_scores_on_grid = test_data
    for step in trange(num_iterations):
        key, subkey = random.split(key)
        data = jax.random.uniform(subkey, (batch_size, dim), minval=-10, maxval=10)
        # data = init_distribution.sample(batch_size)
        # idx = jax.random.choice(subkey, grid_points.shape[0], (batch_size,))

        loss, opt = train_op(opt, data)
        running_avg_loss += loss[0]
        if step % testing_freq == testing_freq - 1:
            f_x = grad(net.apply, argnums=2)
            v_f_x = jax.vmap(f_x, in_axes=[None, None, 0])
            vv_f_x = jax.vmap(v_f_x, in_axes=[None, 0, None])
            scores_pred = vv_f_x(opt.target, time_stamps, grid_points)
            testing_error = jnp.mean(jnp.sum((scores_pred - gaussian_scores_on_grid) ** 2, axis=(2,)))




            # testing_error = []
            # for i, t in enumerate(time_stamps):
            #     score_pred = v_f_x(opt.target, t, grid_points)
            #     loss_t = jnp.mean(jnp.sum((score_pred - gaussian_scores_on_grid[i]) ** 2, axis=(1,)))
            #     testing_error.append(loss_t)
            #     print(t)
            # testing_error = jnp.mean(jnp.array(testing_error))

            print('Step %04d  Loss %.5f  Test %.5f' %     (step + 1,
                                                            running_avg_loss    /   (step + 1),
                                                            testing_error
                                                            )
                  )

    return opt.target






if __name__ == '__main__':
    key = random.PRNGKey(1)
    key, subkey = random.split(key)


    mu0 = jnp.zeros((dim,))
    sigma0 = jnp.eye(dim)
    init_distribution = distribution.Gaussian(mu0, sigma0, subkey)

    key, subkey = random.split(key)
    mu_target = jnp.zeros((dim,)) + 4
    # sigma_target = jax.random.normal(subkey, (dim, dim))
    sigma_target = jnp.eye(dim)
    target_potential = potential.QuadraticPotential(mu_target, sigma_target)

    # construct the NODE
    key, subkey = random.split(key)
    net = DenseNet3(1, subkey)

    # compute the testing data
    test_time_stamps = jnp.linspace(0, T, num=101)
    x, y = jnp.linspace(-10, 10, num=101), jnp.linspace(-10, 10, num=101)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([jnp.reshape(xx, (-1)), jnp.reshape(yy, (-1))], axis=1)
    gaussian_scores_on_grid = eval_Gaussian_score_FP(sigma0, mu0, sigma_target, mu_target, test_time_stamps, grid_points, 1)
    test_data = [test_time_stamps, grid_points, gaussian_scores_on_grid]
    params = train_PINN(net, init_distribution, target_potential, test_data, key)

    # plot_result(net, params, init_distribution, target_potential, T)
