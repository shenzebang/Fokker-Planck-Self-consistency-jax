import jax.numpy as jnp
import jax.random as random
import objax
import jax
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
    # transforms.Normalize((0.1307,), (0.3081,))
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




# net = ScoreNet(512)
key = random.PRNGKey(3)
key, subkey = random.split(key)
net = UNet(marginal_prob_std_fn, subkey)


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


def train_op(x, key):
    g, v = gv(x, key)
    opt(lr, g)
    return v


train_op_jit = objax.Jit(train_op, gv.vars() + opt.vars())
# train_op_jit = train_op
n_epoch = 500
# batch_size = 32
lr = 1e-4


def train(key):
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



def test(key):
    _key = random.PRNGKey(3)
    _key, _subkey = random.split(_key)
    net = UNet(marginal_prob_std_fn, _subkey)
    objax.io.load_var_collection(os.path.join(model_path, 'scorenet.npz'), net.vars())
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


if __name__ == '__main__':
    key = random.PRNGKey(1)
    train(key)
    key = random.PRNGKey(2)
    test(key)
