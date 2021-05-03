import jax.numpy as jnp
from jax.tree_util import tree_structure
import jax.random as random
import objax
from objax.variable import TrainVar, StateVar
import jax
from jax import jvp, grad, value_and_grad
from jax.experimental.ode import odeint
import numpy as np
import torch

from utils import *
from model.neural_ode_model import ScoreNet, UNet
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from sampler import ode_sampler
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import os

# global configuration
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = ".5"
DATA_DIR = os.path.join(os.environ['HOME'], 'DATASET')
model_path = './ckpt'
if not os.path.exists(model_path): os.makedirs(model_path)

batch_size = {
    "train": 32,
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
tolerance = 1e-3


def train(key):
    # key = random.PRNGKey(3)
    key, subkey = random.split(key)
    net = UNet(marginal_prob_std_fn, subkey)

    @objax.Function.with_vars(net.vars())
    def score_matching_loss_fn(x, key):
        key, subkey = random.split(key)
        random_t = random.uniform(subkey, [x.shape[0]])
        key, subkey = random.split(key)
        z = random.normal(subkey, x.shape)
        std = marginal_prob_std_fn(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = net(random_t, perturbed_x, training=True)
        loss = jnp.mean(jnp.sum((score * std[:, None, None, None] + z) ** 2, axis=(1, 2, 3)))
        return loss

    opt = objax.optimizer.Adam(net.vars())

    gv = objax.GradValues(score_matching_loss_fn, net.vars())

    @objax.Function.with_vars(net.vars() + gv.vars() + opt.vars())
    def train_op(x, key):
        g, v = gv(x, key)
        opt(lr, g)
        return v

    # train_op_jit = jax.jit(train_op)

    train_op_jit = objax.Jit(train_op)

    # @objax.Function.with_vars(net.vars() + gv.vars() + opt.vars())
    # def train_op(x, key):
    #     g, v = gv(x, key)
    #     opt(lr, g)
    #     return v

    # train_op_jit = objax.Jit(train_op, net.vars() + opt.vars())
    # train_op_jit = objax.Jit(train_op)
    # for _i, (_k, _v) in enumerate(train_op.vars().items()):
    #     print(_i, _k, _v.shape)
    # print("======================")
    # for i, (k, value) in enumerate(gv.vars().items()):
    #     print(i, k, value.shape)
    # print("======================")
    # print("======================")
    # for i, (k, value) in enumerate(net.vars().items()):
    #     print(i, k, value.shape)
    # print("======================")
    for epoch in trange(n_epoch):
        # training process
        avg_loss = 0.
        num_data = 0
        for _, (data, _) in enumerate(dataloader["train"]):
            key, subkey = random.split(key)
            data = jnp.array(data.numpy())
            _loss = train_op_jit(data, subkey)[0]
            avg_loss += _loss
            num_data += data.shape[0]
        avg_loss /= num_data
        print('Epoch %04d  Loss %.5f' % (epoch + 1, avg_loss))

    objax.io.save_var_collection(os.path.join(model_path, 'scorenet.npz'), net.vars())


num_critic_epoch = 1


def train_critic(key):
    key, subkey = random.split(key)
    net = UNet(marginal_prob_std_fn, subkey)
    critic = UNet(marginal_prob_std_fn, subkey)
    objax.io.load_var_collection(os.path.join(model_path, 'scorenet.npz'), net.vars())
    objax.io.load_var_collection(os.path.join(model_path, 'scorenet.npz'), critic.vars())

    # define gradient and value, no need for adjoint system
    def critic_gv_fn(x_init, key):
        # define v for the Hutchinson’s Estimator
        key, subkey = random.split(key)
        v = random.normal(subkey, tuple([20] + list(x_init.shape)))
        # define the initial states
        t_0 = jnp.zeros(x_init.shape[0])
        score_init = net(t_0, x_init, training=False)
        critic_loss_init = jnp.zeros(1)
        critic_grad_init = [jnp.zeros_like(_var) for _var in critic.vars().subset(is_a=TrainVar)]
        state_init = [x_init, score_init, critic_loss_init, critic_grad_init]

        def ode_func(states, t):
            x = states[0]
            score = states[1]
            _t = jnp.ones([x.shape[0]]) * t
            diffusion_weight = diffusion_coeff_fn(t)
            score_pred = net(_t, x, training=False)
            dx = -.5 * (diffusion_weight ** 2) * score_pred

            f = lambda x: net(_t, x, training=False)

            def divergence_fn(_x, _v):
                # Hutchinson’s Estimator
                # computes the divergence of net at x with random vector v
                _, u = jvp(f, (_x,), (_v,))
                return jnp.sum(jnp.dot(u, _v))

            batch_div_fn = jax.vmap(divergence_fn, in_axes=[None, 0])

            def batch_div(x):
                return batch_div_fn(x, v).mean(axis=0)

            grad_div_fn = grad(batch_div)

            dscore_1 = - grad_div_fn(x)
            dscore_2 = - jvp(f, (x,), (score,))[1]  # f(x), df/dx * v = jvp(f, x, v)
            dscore = dscore_1 + dscore_2

            def dcritic_loss_fn(_x):
                critic_pred = critic(_t, _x, training=True)
                loss = ((critic_pred - score_pred) ** 2).sum(axis=(1, 2, 3)).mean()
                return loss

            dc_gv = objax.GradValues(dcritic_loss_fn, critic.vars())
            # dc_gv = lambda x: (x, x+1)
            dcritic_grad, dcritic_loss = dc_gv(x)
            # dcritic_grad = flatten(dcritic_grad)
            dcritic_loss = dcritic_loss[0][None]
            # print("loss ", dcritic_loss[0].shape, "grad", dcritic_grad.shape)
            dstates = [dx, dscore, dcritic_loss, dcritic_grad]

            # print("============")
            # for _var in dcritic_grad:
            #     print(_var.shape)
            # print("============")
            return dstates

        # ode_func = jax.jit(ode_func)
        tspace = np.array((0., 1.))

        result = odeint(ode_func, state_init, tspace, atol=tolerance, rtol=tolerance)

        _g = [_var[1] for _var in result[3]]
        # print("============")
        # for _var in _g:
        #     print(_var.shape)
        # print("============")
        return _g, result[2][1]

    # define optimizer
    opt = objax.optimizer.Adam(critic.vars())

    # define train_op
    def train_op(x, key):
        # g, v = critic_gv_fn_jit(x, key)
        g, v = critic_gv_fn(x, key)
        # g = _reshape_like(g)
        # change the shape of g!
        # critic.vars().subset(is_a=StateVar).assign(svars_t)
        opt(lr, g)
        return v

    # train_op_jit = jax.jit(train_op)
    # todo: why is this one better?
    train_op = objax.Jit(train_op, critic.vars().subset(is_a=TrainVar) + opt.vars())

    # training process
    for epoch in range(num_critic_epoch):
        avg_loss = 0.
        num_data = 0
        for step, (data, _) in enumerate(dataloader["train"]):
            key, subkey = random.split(key)
            data = jnp.array(data.numpy())
            _loss = train_op(data, subkey)
            # print(type(svars_t))
            # _loss = train_op(data, subkey)[0]
            avg_loss += _loss[0]
            num_data += data.shape[0]
            print("Step %04d  Loss %.5f" % (step, avg_loss / num_data))
            if step > 0:
                break
            # for _k, _var in critic.vars().subset(is_a=StateVar).items():
            #     print(_k, _var)
        for _k, _var in critic.vars().subset(is_a=StateVar).items():
            print(_k, _var)
        avg_loss /= num_data
        print('Epoch %04d  Loss %.5f' % (epoch + 1, avg_loss))
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
    return critic

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
                return jnp.mean(jnp.sum((actor(_t, _x, training=True) + critic(_t, _x, training=False)) ** 2, axis=(1,2,3)))

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



def test(key, ckpt_file):
    _key = random.PRNGKey(3)
    _key, _subkey = random.split(_key)
    net = UNet(marginal_prob_std_fn, _subkey)
    objax.io.load_var_collection(ckpt_file, net.vars())
    ## Load the pre-trained checkpoint from disk.

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


def test2(key, ckpt_file, new_tensor):
    _key = random.PRNGKey(3)
    _key, _subkey = random.split(_key)
    net = UNet(marginal_prob_std_fn, _subkey)
    objax.io.load_var_collection(ckpt_file, net.vars())
    ## Load the pre-trained checkpoint from disk.
    net.vars().subset(is_a=TrainVar).assign(new_tensor)

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
    key = random.PRNGKey(1)
    train(key)
    key = random.PRNGKey(1)
    critic = train_critic(key)
    objax.io.save_var_collection(os.path.join(model_path, 'critic_train.npz'), critic.vars().subset(is_a=TrainVar))
    objax.io.save_var_collection(os.path.join(model_path, 'critic_state.npz'), critic.vars().subset(is_a=StateVar))
    # key = random.PRNGKey(2)
    # test_critic(key, critic)
    ckpt_file = os.path.join(model_path, 'scorenet.npz')
    # ckpt_file = os.path.join(model_path, 'critic.npz')
    test(key, ckpt_file)
    test2(key, ckpt_file, critic.vars().subset(is_a=TrainVar).tensors())
