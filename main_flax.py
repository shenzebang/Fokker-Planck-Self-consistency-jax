import flax.optim
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import jax.random as random
from flax import serialization

import jax
from jax import jvp, grad, value_and_grad, vjp
from jax.experimental.ode import odeint
import numpy as np
import torch

from utils import *
from model.neural_ode_model_flax import UNet
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from sampler import ode_sampler
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import os
import functools

# global configuration
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = ".5"
DATA_DIR = os.path.join(os.environ['HOME'], 'DATASET')
model_path = './ckpt'
if not os.path.exists(model_path): os.makedirs(model_path)

batch_size = {
    "train": 8,
    "test": 1024
}
num_workers = 8

# load dataset with PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = {
    "train": MNIST(os.path.join(DATA_DIR, "mnist"), train=True, transform=transform, download=True),
    "test": MNIST(os.path.join(DATA_DIR, "mnist"), train=False, transform=transform, download=True)
}
dataloader = {
    "train": DataLoader(dataset["train"], batch_size=batch_size["train"], shuffle=True, num_workers=num_workers),
    "test": DataLoader(dataset["test"], batch_size=batch_size["test"], shuffle=False, num_workers=num_workers)
}

sigma = 25.
marginal_prob_std_fn = lambda t: marginal_prob_std(t, sigma)
diffusion_coeff_fn = lambda t: diffusion_coeff(t, sigma)

# train_op_jit = train_op
n_epoch = 500
# batch_size = 32
lr = 1e-4
tolerance = 1e-5


def train(key):
    # key = random.PRNGKey(3)
    key, subkey = random.split(key)
    net = UNet(subkey)
    key, subkey = random.split(key)
    params = net.init(subkey, jnp.zeros(1), jnp.zeros((1, 28, 28, 1)))

    def score_matching_loss_fn(params, x, key):
        key, subkey = random.split(key)
        random_t = random.uniform(subkey, [x.shape[0]])
        key, subkey = random.split(key)
        z = random.normal(subkey, x.shape)
        std = marginal_prob_std_fn(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = net.apply(params, random_t, perturbed_x)
        loss = jnp.mean(jnp.sum((score * std[:, None, None, None] + z) ** 2, axis=(1, 2, 3)))
        return loss

    opt_def = flax.optim.Adam(learning_rate=lr)
    opt = opt_def.create(params)
    # score_matching_loss_fn = jax.jit(score_matching_loss_fn)
    loss_grad_fn = jax.value_and_grad(score_matching_loss_fn)

    def train_op(x, opt, key):
        v, g = loss_grad_fn(opt.target, x, key)
        return v, opt.apply_gradient(g)

    train_op = jax.jit(train_op)

    for epoch in trange(n_epoch):
        # training process
        avg_loss = 0.
        num_data = 0
        for _, (data, _) in enumerate(dataloader["train"]):
            key, subkey = random.split(key)
            data = jnp.array(data.numpy())
            data = jnp.transpose(data, axes=(0, 2, 3, 1))
            # print(data.shape)
            _loss, opt = train_op(data, opt, subkey)
            # print(_loss)
            avg_loss += _loss
            num_data += data.shape[0]
        avg_loss /= num_data
        print('Epoch %04d  Loss %.5f' % (epoch + 1, avg_loss))

    # objax.io.save_var_collection(os.path.join(model_path, 'scorenet.npz'), net.vars())
    dict_output = serialization.to_state_dict(opt.target)
    jnp.save(os.path.join(model_path, 'scorenet.npy'), [dict_output])
    # return opt.target


num_critic_epoch = 1


def train_critic(key, critic_params, actor):
    key, subkey = random.split(key)
    net = UNet(subkey)
    # actor = lambda t, x: net.apply(actor_params, t, x)
    # critic_params = net.init(subkey, jnp.zeros(1), jnp.zeros((1, 28, 28, 1)))
    critic_flat, critic_tree = tree_flatten(critic_params)
    f = lambda t, x: -.5 * (diffusion_coeff_fn(t) ** 2) * actor(t, x)

    # define gradient and value, no need for adjoint system
    def critic_gv_fn(params, x_init, key):
        # define v for the Hutchinson’s Estimator
        key, subkey = random.split(key)
        v = random.normal(subkey, tuple([50] + list(x_init.shape)))
        # define the initial states
        score_init = actor(jnp.zeros(x_init.shape[0]), x_init)
        critic_loss_init = jnp.zeros(1)
        net_loss_init = jnp.zeros(1)
        critic_grad_init = [jnp.zeros_like(_var) for _var in critic_flat]
        state_init = [x_init, score_init, critic_loss_init, critic_grad_init, net_loss_init]

        # critic = lambda t, x: net.apply(_critic_params, t, x)
        def ode_func(states, t):
            x = states[0]
            score = states[1]
            f_t = lambda x: f(t, x)

            dx = f_t(x)

            div_f_t = lambda x, v: divergence_fn(f_t, x, v)

            grad_div_fn = grad(div_f_t)

            dscore_1 = - grad_div_fn(x, v)
            # dscore_2 = - jvp(f_t, (x,), (score,))[1]  # f(x), df/dx * v = jvp(f, x, v)
            _, vjp_fn = vjp(f_t, x)
            dscore_2 = - vjp_fn(score)[0]
            # print(len(dscore_2), type(dscore_2[0]))
            dscore = dscore_1 + dscore_2

            weight_loss = .0001 ** t
            weight_reg = 1. ** t
            def dcritic_loss_fn(_critic_params, _x):
                critic_pred = net.apply(_critic_params, t, _x)
                loss = ((critic_pred - score) ** 2).sum(axis=(1, 2, 3)).mean() * weight_loss
                reg = (critic_pred ** 2).sum(axis=(1, 2, 3)).mean() * weight_reg
                return loss + reg

            dc_vg = jax.value_and_grad(dcritic_loss_fn)

            dcritic_loss, dcritic_grad = dc_vg(params, x)

            dcritic_loss = dcritic_loss[None]

            score_pred = actor(t, x)
            dnet_loss = ((score_pred - score) ** 2).sum(axis=(1, 2, 3)).mean()[None] * weight_loss

            dstates = [dx, dscore, dcritic_loss, dcritic_grad, dnet_loss]

            return dstates

        tspace = np.array((0., 1.))

        result = odeint(ode_func, state_init, tspace, atol=tolerance, rtol=tolerance)

        _g = tree_unflatten(critic_tree, [_var[1] for _var in result[3]])
        # weight = 10.
        # def critic_loss_2(_critic_params):
        #     critic_pred = net.apply(_critic_params, jnp.zeros(x_init.shape[0]), x_init)
        #     return jnp.mean(jnp.sum((critic_pred - score_init) ** 2, axis=(1, 2, 3))) * weight

        # vg2 = value_and_grad(critic_loss_2)
        # _, _g2 = vg2(critic_params)

        # return _g, _g2, result[2][1]

        return _g, result[2][1], result[4][1]

    # define optimizer
    opt_def = flax.optim.Adam(learning_rate=lr)
    opt = opt_def.create(critic_params)

    # define train_op
    def train_op(_opt, x, key):
        # g, g2, v = critic_gv_fn(_opt.target, _actor_params, x, key)\
        # return v, _opt.apply_gradient(g).apply_gradient(g2)
        g, v_critic, v_actor = critic_gv_fn(_opt.target, x, key)
        return v_critic, _opt.apply_gradient(g), v_actor

    train_op = jax.jit(train_op)

    # training process
    for epoch in range(num_critic_epoch):
        avg_loss = 0.
        # num_data = 0
        for step, (data, _) in enumerate(dataloader["train"]):
            key, subkey = random.split(key)
            data = jnp.array(data.numpy())
            data = jnp.transpose(data, axes=(0, 2, 3, 1))
            _loss_critic, opt, _loss_actor = train_op(opt, data, subkey)
            avg_loss += _loss_critic[0]
            # num_data += data.shape[0]
            print("Step %04d  Loss %.5f Loss(net) %.5f" % (step, _loss_critic, _loss_actor))
            if step >= 100:
                break
        avg_loss /= step
        print('Epoch %04d  Loss %.5f' % (epoch + 1, avg_loss))

    dict_output = serialization.to_state_dict(opt.target)
    # jnp.save(os.path.join(model_path, 'critic.npy'), [dict_output])
    # print(type(critic.vars().subset(is_a=TrainVar).tensors()[0]))
    # np.save(os.path.join(model_path, 'critic1.npy'), np.array(critic.vars().tensors()[0]))
    # vars = []
    # print("length is ", len(critic.vars().tensors()))
    # for index, var in enumerate(critic.vars().tensors()):
    #     print(index)
    #     vars.append(np.array(var))
    #
    # np.savez(os.path.join(model_path, 'critic2.npz'), vars)
    # with jax.disable_jit():
    #     objax.io.save_var_collection(os.path.join(model_path, 'critic.npz'), critic.vars())
    # return critic
    return opt.target


def compute_critic_loss(key, net_params):
    key, subkey = random.split(key)
    net = UNet(subkey)
    # critic_params = net.init(subkey, jnp.zeros(1), jnp.zeros((1, 28, 28, 1)))
    net_flat, net_tree = tree_flatten(net_params)

    # define gradient and value, no need for adjoint system
    def critic_v_fn(_net_params, x_init, key):
        # define v for the Hutchinson’s Estimator
        key, subkey = random.split(key)
        v = random.normal(subkey, tuple([50] + list(x_init.shape)))
        # define the initial states
        score_init = net.apply(_net_params, jnp.zeros(x_init.shape[0]) + 1e-3, x_init)
        critic_loss_init = jnp.zeros(1)
        state_init = [x_init, score_init, critic_loss_init]

        def ode_func(states, t):
            x = states[0]
            score = states[1]
            _t = jnp.ones([x.shape[0]]) * t
            diffusion_weight = diffusion_coeff_fn(t)
            score_pred = net.apply(_net_params, _t, x)
            dx = -.5 * (diffusion_weight ** 2) * score_pred

            f = lambda x: -.5 * (diffusion_weight ** 2) * net.apply(_net_params, _t, x)

            def divergence_fn(_x, _v):
                # Hutchinson’s Estimator
                # computes the divergence of net at x with random vector v
                _, u = jvp(f, (_x,), (_v,))
                # print(u.shape, _x.shape, _v.shape)
                return jnp.sum(u * _v)

            batch_div_fn = jax.vmap(divergence_fn, in_axes=[None, 0])

            def batch_div(x):
                return batch_div_fn(x, v).mean(axis=0)

            grad_div_fn = grad(batch_div)

            dscore_1 = - grad_div_fn(x)
            dscore_2 = - jvp(f, (x,), (score,))[1]  # f(x), df/dx * v = jvp(f, x, v)
            dscore = dscore_1 + dscore_2

            dcritic_loss = ((score_pred - score) ** 2).sum(axis=(1, 2, 3)).mean()

            dcritic_loss = dcritic_loss[None]

            dstates = [dx, dscore, dcritic_loss]

            return dstates

        tspace = np.array((0., 1.))

        result = odeint(ode_func, state_init, tspace, atol=tolerance, rtol=tolerance)

        return result[2][1]

    critic_v_fn = jax.jit(critic_v_fn)

    # training process
    for epoch in range(num_critic_epoch):
        avg_loss = 0.
        num_data = 0
        for step, (data, label) in enumerate(dataloader["train"]):
            key, subkey = random.split(key)
            print(label)
            data = jnp.array(data.numpy())
            data = jnp.transpose(data, axes=(0, 2, 3, 1))
            _loss = critic_v_fn(net_params, data, subkey)
            avg_loss += _loss[0]
            num_data += data.shape[0]
            print("Step %04d  Loss %.5f" % (step, avg_loss / num_data))
        avg_loss /= num_data
        print('Epoch %04d  Loss %.5f' % (epoch + 1, avg_loss))


num_actor_epoch = 10


def train_actor(key):
    # initialize the actor model
    key, subkey = random.split(key)
    actor = UNet(marginal_prob_std_fn, subkey)
    objax.io.load_var_collection(os.path.join(model_path, 'actor.npz'), actor.vars())
    # initialize the critic model
    critic = UNet(marginal_prob_std_fn, subkey)
    objax.io.load_var_collection(os.path.join(model_path, 'critic.npz'), critic.vars())

    # define the gradient operator using adjoint system
    def g_actor_fn(data):
        # compute x(T) by solve IVP (I) & compute the actor loss
        # ================ Forward ===================
        states_init = [data, jnp.zeros(1)]

        def ode_func1(states, t):
            x = states[0]
            _t = jnp.ones([x.shape[0]]) * t
            dx = actor(_t, x, training=True)
            dx_critic = critic(_t, x, training=False)
            dloss = jnp.mean(jnp.sum((dx + dx_critic) ** 2, axis=(1, 2, 3)))
            return [dx, dloss]

        tspace = np.array((0., 1.))

        result = odeint(ode_func1, states_init, tspace, atol=tolerance, rtol=tolerance)
        loss_actor = result[0][1]
        x_T = result[1][1]
        # ================ Forward ===================
        # ================ Backward ==================
        # compute dl/d theta via adjoint method
        a_T = jnp.zeros_like(x_T)
        actor_g = [jnp.zeros_like(_var) for _var in actor.vars().subset(is_a=TrainVar)]
        states_init = [x_T, a_T, actor_g]

        def ode_func2(states, t):
            x = states[0]
            a = states[1]
            _t = jnp.ones([x.shape[0]]) * (1. - t)
            dx = actor(_t, x, training=True)

            def _f(_x):
                _actor_x = actor(_t, _x, training=True)
                _f1 = jnp.dot(_actor_x, a)
                _critic_x = critic(_t, _x, training=False)
                _f2 = jnp.mean(jnp.sum((_actor_x + _critic_x) ** 2, axis=(1, 2, 3)))
                return -(_f1 + _f2)

            da_fn = jax.grad(_f)
            da = da_fn(x)

            def _g(_x):
                return jnp.mean(
                    jnp.sum((actor(_t, _x, training=True) + critic(_t, _x, training=False)) ** 2, axis=(1, 2, 3)))

            g_theta_1 = objax.Grad(lambda x: jnp.dot(a, actor(_t, x, training=True)), actor.vars())
            g_theta_2 = objax.Grad(_g, actor.vars())
            dg_theta = g_theta_1(x) + g_theta_2(x)

            return [-dx, -da, -dg_theta]

        # ================ Backward ==================

        result = odeint(ode_func2, states_init, tspace, atol=tolerance, rtol=tolerance)
        return result[2][1], loss_actor

    # define the optimizer
    opt = objax.optimizer.Adam(critic.vars())

    # define train_op
    def train_op(x):
        g, v, = g_actor_fn(x)
        opt(lr, g)
        return v

    train_op = objax.Jit(train_op, actor.vars().subset(is_a=TrainVar) + opt.vars())

    # training process
    for epoch in range(num_actor_epoch):
        avg_loss = 0.
        num_data = 0
        for step, (data, _) in enumerate(dataloader["train"]):
            data = jnp.array(data.numpy())
            _loss = train_op(data)
            avg_loss += _loss[0]
            num_data += data.shape[0]
            print("Step %04d  Loss %.5f" % (step, avg_loss / num_data))
            # if step > 100:
            #     break
        avg_loss /= num_data
        print('Epoch %04d  Loss %.5f' % (epoch + 1, avg_loss))

    return actor


def train_nwgf(key, actor_params):
    key, subkey = random.split(key)
    net = UNet(subkey)
    # actor_params = net.init(subkey, jnp.zeros(1), jnp.zeros((1, 28, 28, 1)))

    params_flat, params_tree = tree_flatten(actor_params)

    def nwgf_gv(params, data, key):
        key, subkey = random.split(key)
        # stochastic vectors used for trace (divergence) estimation
        v = random.normal(subkey, tuple([100] + list(data.shape)))
        f = lambda _x, _t, _params: net.apply(_params, _t, _x) * -.5 * (diffusion_coeff_fn(_t) ** 2)
        # compute x(T) by solve IVP (I) & compute the actor loss
        # ================ Forward ===================
        x_0 = data
        xi_0 = net.apply(params, jnp.zeros(1), x_0)
        loss_0 = jnp.zeros(1)
        states_0 = [x_0, xi_0, loss_0]
        def ode_func1(states, t):
            x = states[0]
            xi = states[1]
            f_t_theta = lambda _x: f(_x, t, params)
            dx = f_t_theta(x)

            def h_t_theta(in_1, in_2):
                # in_1 is xi
                # in_2 is x
                # in_3 is theta
                div_f_t_theta = lambda _x: divergence_fn(f_t_theta, _x, v)
                grad_div_fn = grad(div_f_t_theta)
                h1 = - grad_div_fn(in_2)
                _, vjp_fn = vjp(f_t_theta, in_2)
                h2 = - vjp_fn(in_1)[0]
                return h1 + h2
            dxi = h_t_theta(xi, x)
            weight_t = (1 ** t) / sigma**3
            def g_t(in_1, in_2):
                # in_1 is xi
                # in_2 is x
                return jnp.mean(jnp.sum((f_t_theta(in_2) + in_1 * .5 * (diffusion_coeff_fn(t) ** 2)) ** 2, axis=(1, 2, 3))) * weight_t

            dloss = g_t(xi, x)

            return [dx, dxi, dloss]
        tspace = np.array((0., 1.))
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
            t = 1. - t
            x = states[0]
            a = states[1]
            b = states[2]
            xi = states[3]


            f_t = lambda _x, _params: f(_x, t, _params)
            dx = f_t(x, params)

            _, vjp_fx_fn = vjp(lambda _x: f_t(_x, params), x)
            vjp_fx_a = vjp_fx_fn(a)[0]
            _, vjp_ftheta_fn = vjp(lambda _params: f_t(x, _params), params)
            vjp_ftheta_a = vjp_ftheta_fn(a)[0]
            def h_t(in_1, in_2, in_3):
                # in_1 is xi
                # in_2 is x
                # in_3 is theta
                f_t_theta = lambda _x: f_t(_x, in_3)
                div_f_t_theta = lambda _x: divergence_fn(f_t_theta, _x, v)
                grad_div_fn = grad(div_f_t_theta)
                h1 = - grad_div_fn(in_2)
                _, vjp_fn = vjp(f_t_theta, in_2)
                h2 = - vjp_fn(in_1)[0]
                return h1 + h2
            _, vjp_hxi_fn = vjp(lambda _xi: h_t(_xi, x, params), xi)
            vjp_hxi_b = vjp_hxi_fn(b)[0]
            _, vjp_hx_fn = vjp(lambda _x: h_t(xi, _x, params), x)
            vjp_hx_b = vjp_hx_fn(b)[0]
            _, vjp_htheta_fn = vjp(lambda _params: h_t(xi, x, _params), params)
            vjp_htheta_b = vjp_htheta_fn(b)[0]
            weight_t = (1. ** t) / sigma**3
            def g_t(in_1, in_2, in_3):
                # in_1 is xi
                # in_2 is x
                # in_3 is theta
                return jnp.mean(jnp.sum((f_t(in_2, in_3) + in_1 * .5 * (diffusion_coeff_fn(t) ** 2)) ** 2, axis=(1, 2, 3))) * weight_t
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
            dgrad = [_dgrad1 + _dgrad2 + _dgrad3 for _dgrad1, _dgrad2, _dgrad3 in zip(vjp_ftheta_a_flat, vjp_htheta_b_flat, dthetag_flat)]
            # dgrad = vjp_ftheta_a + vjp_htheta_b + dthetag(xi, x, params)

            return [-dx, -da, -db, -dxi, dloss, dgrad]
        # ================ Backward ==================
        tspace = np.array((0., 1.))
        result_backward = odeint(ode_func2, states_T, tspace, atol=tolerance, rtol=tolerance)

        grad_T = tree_unflatten(params_tree, [_var[1] for _var in result_backward[5]])
        x_0_b = result_backward[0][1]
        xi_0_b = result_backward[3][1]
        error_x = jnp.sum((x_0_b - x_0)**2)
        error_xi = jnp.sum((xi_0 - xi_0_b)**2)

        loss_b = result_backward[4][1]
        return grad_T, loss_b, loss_f, error_x, error_xi

    opt_def = flax.optim.Adam(learning_rate=lr)
    opt = opt_def.create(actor_params)
    nwgf_gv = jax.jit(nwgf_gv)
    # define train_op
    def train_op(_opt, x, key):
        # g, g2, v = critic_gv_fn(_opt.target, _actor_params, x, key)\
        # return v, _opt.apply_gradient(g).apply_gradient(g2)
        g, v_b, v_f, error_x, error_xi = nwgf_gv(_opt.target, x, key)
        print(error_x, error_xi)
        return v_b, _opt.apply_gradient(g), v_f

    # train_op = jax.jit(train_op)

    # training process
    for epoch in range(num_critic_epoch):
        avg_loss = 0.
        # num_data = 0
        step = 0
        for _, (data, _) in enumerate(dataloader["train"]):
            step += 1
            key, subkey = random.split(key)
            data = jnp.array(data.numpy())
            data = jnp.transpose(data, axes=(0, 2, 3, 1))
            loss_b, opt, loss_f = train_op(opt, data, subkey)
            avg_loss += loss_b[0]
            print("Step %04d  Loss B %.5f Loss F %.5f" % (step, loss_b, loss_f))
            if step >= 10:
                break
        avg_loss /= step
        print('Epoch %04d  Loss %.5f' % (epoch + 1, avg_loss))

    return opt.target
# def test(key, params):
#     _key = random.PRNGKey(3)
#     _key, _subkey = random.split(_key)
#     net = UNet(_subkey)
#     ## Load the pre-trained checkpoint from disk.
#     # opt_def = flax.optim.Adam(learning_rate=lr)
#     # opt = opt_def.create(params)
#
#
#     sample_batch_size = 64  # @param {'type':'integer'}
#     sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
#
#     ## Generate samples using the specified sampler.
#     key, subkey = random.split(key)
#     samples = sampler(net,
#                       params,
#                       marginal_prob_std_fn,
#                       diffusion_coeff_fn,
#                       subkey,
#                       sample_batch_size,
#                       eps=1e-3)
#
#     ## Sample visualization.
#     print(jnp.max(samples))
#     samples = jax.lax.clamp(0.0, samples, 1.0)
#     samples = np.array(samples)
#     sample_grid = make_grid(torch.from_numpy(samples), nrow=int(np.sqrt(sample_batch_size)))
#
#     plt.figure(figsize=(6, 6))
#     plt.axis('off')
#     plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
#     plt.show()

# def test_objax(key, ckpt_file):
#     _key = random.PRNGKey(3)
#     _key, _subkey = random.split(_key)
#     net = UNet(marginal_prob_std_fn, _subkey)
#     objax.io.load_var_collection(ckpt_file, net.vars())
#     ## Load the pre-trained checkpoint from disk.
#
#     sample_batch_size = 64  # @param {'type':'integer'}
#     sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
#
#     ## Generate samples using the specified sampler.
#     key, subkey = random.split(key)
#     samples = sampler(net,
#                       marginal_prob_std_fn,
#                       diffusion_coeff_fn,
#                       subkey,
#                       sample_batch_size,
#                       eps=1e-3)
#
#     ## Sample visualization.
#     print(jnp.max(samples))
#     samples = jax.lax.clamp(0.0, samples, 1.0)
#     samples = np.array(samples)
#     sample_grid = make_grid(torch.from_numpy(samples), nrow=int(np.sqrt(sample_batch_size)))
#
#     plt.figure(figsize=(6, 6))
#     plt.axis('off')
#     plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
#     plt.show()

# def test2(key, ckpt_file, new_tensor):
#     _key = random.PRNGKey(3)
#     _key, _subkey = random.split(_key)
#     net = UNet(marginal_prob_std_fn, _subkey)
#     objax.io.load_var_collection(ckpt_file, net.vars())
#     ## Load the pre-trained checkpoint from disk.
#     net.vars().subset(is_a=TrainVar).assign(new_tensor)
#
#     sample_batch_size = 64  # @param {'type':'integer'}
#     sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
#
#     ## Generate samples using the specified sampler.
#     key, subkey = random.split(key)
#     samples = sampler(net,
#                       marginal_prob_std_fn,
#                       diffusion_coeff_fn,
#                       subkey,
#                       sample_batch_size,
#                       eps=1e-3)
#
#     ## Sample visualization.
#     print(jnp.max(samples))
#     samples = jax.lax.clamp(0.0, samples, 1.0)
#     samples = np.array(samples)
#     sample_grid = make_grid(torch.from_numpy(samples), nrow=int(np.sqrt(sample_batch_size)))
#
#     plt.figure(figsize=(6, 6))
#     plt.axis('off')
#     plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
#     plt.show()

def test(key, net):
    sample_batch_size = 64  # @param {'type':'integer'}
    sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
    ## Generate samples using the specified sampler.
    # key, subkey = random.split(key)

    samples = sampler(net,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      key,
                      sample_batch_size,
                      eps=1e-3)

    ## Sample visualization.
    print(jnp.max(samples))
    samples = jax.lax.clamp(0.0, samples, 1.0)
    samples = np.array(samples)
    sample_grid = make_grid(torch.from_numpy(samples), nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()


def test_critic(key, critic):
    net = critic

    sample_batch_size = 64  # @param {'type':'integer'}
    sampler = ode_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    ## Generate samples using the specified sampler.
    key, subkey = random.split(key)
    samples = sampler(net,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      subkey,
                      sample_batch_size,
                      eps=1e-3)

    ## Sample visualization.
    print(jnp.max(samples))
    samples = jax.lax.clamp(0.0, samples, 1.0)
    samples = np.array(samples)
    sample_grid = make_grid(torch.from_numpy(samples), nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()


if __name__ == '__main__':
    for step, (data, _) in enumerate(dataloader["train"]):
        continue
    key = random.PRNGKey(1)
    # train(key)
    # key = random.PRNGKey(1)
    ckpt_file = os.path.join(model_path, 'scorenet.npy')
    scorenet_params = jnp.load(ckpt_file, allow_pickle=True)[0]
    # compute_critic_loss(key, scorenet_params)
    ckpt_file = os.path.join(model_path, 'critic.npy')
    critic_params = jnp.load(ckpt_file, allow_pickle=True)[0]
    key, subkey = random.split(key)
    _net = UNet(subkey)
    actor = lambda t, x: _net.apply(scorenet_params, t, x)
    # critic_params = train_critic(key, critic_params, actor)
    params = train_nwgf(key, scorenet_params)
    actor_nwgf = lambda t, x: _net.apply(params, t, x)
    test(key, actor_nwgf)
    # objax.io.save_var_collection(os.path.join(model_path, 'critic_train.npz'), critic.vars().subset(is_a=TrainVar))
    # objax.io.save_var_collection(os.path.join(model_path, 'critic_state.npz'), critic.vars().subset(is_a=StateVar))

    # test_critic(key, critic)
    # ckpt_file = os.path.join(model_path, 'scorenet.npy')
    # # ckpt_file = os.path.join(model_path, 'critic.npz')
    # key, subkey = random.split(key)
    # net = UNet(subkey)

    # _params = net.init(subkey, jnp.zeros(1), jnp.zeros((1, 28, 28, 1)))
    # params = jnp.load(ckpt_file, allow_pickle=True)[0]
    # params = flax.serialization.from_state_dict(jnp.load(ckpt_file), _params)
    # key = random.PRNGKey(2)
    # key, subkey = random.split(key)
    # _net = UNet(subkey)
    # critic = lambda t, x: _net.apply(critic_params, t, x)
    # test(key, actor)
    # test(key, critic)
    # test(key, lambda t, x: _net.apply(scorenet_params, t, x))
