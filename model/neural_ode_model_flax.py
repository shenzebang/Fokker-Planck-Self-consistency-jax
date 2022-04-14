import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import List


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    embed_dim: int
    key: jnp.ndarray
    scale: float = 30.

    @nn.compact
    def __call__(self, x):
        def _init(key, embed_dim, scale):
            W = jax.random.normal(key, [embed_dim // 2])
            W = W * scale
            return W

        kernel = self.param('random_feature', _init, self.embed_dim, self.scale)
        x_proj = x[:, None] * kernel[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


# class Dense(nn.Module):
#     """A fully connected layer that reshapes outputs to feature maps."""
#     features: int
#     def setup(self):
#         self.dense = nn.Dense(features=self.features)
#
#     def __call__(self, x):
#         x = self.dense.apply(x)
#         return x[..., None, None]

class UNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""
    key: jnp.ndarray
    channels: jnp.array = jnp.array([32, 64, 128, 256])
    embed_dim: int = 256

    def setup(self):
        """Initialize a time-dependent score-based network.
        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        # Gaussian random feature embedding layer for time
        self.embed = GaussianFourierProjection(embed_dim=self.embed_dim, key=self.key)
        self.embed_dense = nn.Dense(features=self.embed_dim)
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1),
                             padding=((1, 1), (1, 1)), use_bias=False)
        self.dense1 = nn.Dense(features=32)
        self.gnorm1 = nn.GroupNorm(num_groups=4)
        # self.gnorm1 = nn.BatchNorm()
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2),
                             padding=((1, 1), (1, 1)), use_bias=False)
        self.dense2 = nn.Dense(features=64)
        self.gnorm2 = nn.GroupNorm(num_groups=32)
        # self.gnorm2 = nn.BatchNorm()
        self.conv3 = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2),
                             padding=((1, 1), (1, 1)), use_bias=False)
        self.dense3 = nn.Dense(features=128)
        self.gnorm3 = nn.GroupNorm(num_groups=32)
        # self.gnorm3 = nn.BatchNorm()
        self.conv4 = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2),
                             padding=((0, 0), (0, 0)), use_bias=False)
        self.dense4 = nn.Dense(features=256)
        self.gnorm4 = nn.GroupNorm(num_groups=32)
        # self.gnorm4 = nn.BatchNorm()

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(2, 2),
                                       padding='VALID', use_bias=False)
        self.dense5 = nn.Dense(features=128)
        self.tgnorm4 = nn.GroupNorm(num_groups=32)
        # self.tgnorm4 = nn.BatchNorm()
        self.tconv3 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2),
                                       padding='SAME', use_bias=False)
        self.dense6 = nn.Dense(64)
        self.tgnorm3 = nn.GroupNorm(num_groups=32)
        # self.tgnorm3 = nn.BatchNorm()
        self.tconv2 = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2),
                                       padding='SAME', use_bias=False)
        self.dense7 = nn.Dense(32)
        self.tgnorm2 = nn.GroupNorm(num_groups=32)
        # self.tgnorm2 = nn.BatchNorm()
        self.tconv1 = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                       use_bias=True)

        # The swish activation function
        self.act = lambda x: x * jax.nn.sigmoid(x)

    def __call__(self, t, x):
        # Obtain the Gaussian random feature embedding for t
        if type(t) is float:
            t = jnp.ones([x.shape[0]]) * t
        elif t.ndim == 0 or t.shape[0] == 1:
            t = jnp.ones([x.shape[0]]) * t
        embed = self.act(self.embed_dense(self.embed(t)))
        # print("embed.shape", embed.shape)
        # print("x.shape", x.shape)
        # print(t.shape)
        # assert not torch.isnan(torch.sum(embed))
        # Encoding path

        h1 = self.conv1(x)
        # print("h1.shape after conv1", h1.shape)
        ## Incorporate information from t
        h1 += self.dense1(embed)[:, None, None, :]
        # print("dense1 shape", self.dense1(embed).shape)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        # print(h1.shape)
        h2 = self.conv2(h1)
        # print(h2.shape)
        h2 += self.dense2(embed)[:, None, None, :]
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)[:, None, None, :]
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)[:, None, None, :]
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        # print(x.shape)
        # print(h1.shape)
        # print(h2.shape)
        # print(h3.shape)
        # print(h4.shape)
        # Decoding path
        h = self.tconv4(h4)
        # print(h.shape)
        ## Skip connection from the encoding path
        h += self.dense5(embed)[:, None, None, :]
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(jnp.concatenate([h, h3], axis=3))
        # print(h.shape)

        h += self.dense6(embed)[:, None, None, :]
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(jnp.concatenate([h, h2], axis=3))
        # print(h.shape)

        h += self.dense7(embed)[:, None, None, :]
        h = self.tgnorm2(h)
        h = self.act(h)
        # print(h.shape)

        h = self.tconv1(jnp.concatenate([h, h1], axis=3))
        # print(h.shape)

        # assert not torch.isnan(torch.sum(h))

        # Normalize output
        # h = h / self.marginal_prob_std(t)[:, None, None, None]
        # print(self.marginal_prob_std(t))
        # assert not torch.isnan(torch.sum(h))
        return h


class ConcatSquashLinear(nn.Module):
    dim_out: int = 64

    def setup(self):
        self._layer = nn.Dense(features=self.dim_out)
        self._hyper_bias = nn.Dense(features=self.dim_out, use_bias=False)
        self._hyper_gate = nn.Dense(features=self.dim_out)

    def __call__(self, t, x):
        if type(t) is float:
            t = jnp.ones(1) * t
        elif t.ndim == 0:
            t = jnp.ones(1) * t

        return self._layer(x) * jax.nn.sigmoid(self._hyper_gate(t)) \
               + self._hyper_bias(t)


class DenseNet(nn.Module):
    dim: int  # the ambient dimension of the problem
    hidden_dims = [64, 64, 64]

    def setup(self):
        self.layers = [ConcatSquashLinear(dim_out) for dim_out in self.hidden_dims + [self.dim]]

        # for dim_out in self.hidden_dims + [self.dim]:
        #     layer = ConcatSquashLinear(dim_out)
        #     self.layers.append(layer)

        self.activation = jax.nn.tanh

    def __call__(self, t, x):
        if type(t) is float:
            t = jnp.ones(1) * t
        elif t.ndim == 0:
            t = jnp.ones(1) * t

        dx = x
        for i, layer in enumerate(self.layers):
            dx = layer(t, dx)
            if i < len(self.layers) - 1:
                dx = self.activation(dx)

        return dx

class G2GNet(nn.Module):
    dim: int
    mu0_offset: float
    sigma_0: float
    def setup(self):
        self.layer = nn.Dense(features=self.dim, use_bias=False)
        self.layer_sigma = nn.Dense(features=1, use_bias=False)


    def __call__(self, t, x):
        if type(t) is float:
            t = jnp.ones(1) * t
        elif t.ndim == 0:
            t = jnp.ones(1) * t
        mu_t = self.layer(jnp.ones(1)) * jnp.exp(-t)
        sigma_t_2 = self.layer_sigma(jnp.ones(1)) * jnp.exp(-t*2) + 1
        # sigma_t_2 = (self.sigma_0 ** 2 - 1) * jnp.exp(-t * 2) + 1
        # mu_t = jnp.ones(x.shape[-1]) * self.mu0_offset * jnp.exp(-t)
        return (x - mu_t)/sigma_t_2




class DenseNet2(nn.Module):
    dim: int  # the ambient dimension of the problem
    key: jnp.ndarray
    hidden_dims = [64, 64, 64]
    # hidden_dims = [64, 64, 64]
    embed_dim: int = 64


    def setup(self):
        self.embed = GaussianFourierProjection(embed_dim=self.embed_dim, key=self.key)
        self.embed_dense = nn.Dense(features=self.embed_dim)
        self.layers = [nn.Dense(dim_out) for dim_out in self.hidden_dims + [self.dim]]
        self.embed_denses = [nn.Dense(dim_out) for dim_out in self.hidden_dims + [self.dim]]
        # for dim_out in self.hidden_dims + [self.dim]:
        #     layer = ConcatSquashLinear(dim_out)
        #     self.layers.append(layer)
        self.act = lambda x: x * jax.nn.sigmoid(x)


    def __call__(self, t, x):


        if type(t) is float:
            t = jnp.ones(1) * t
        elif t.ndim == 0:
            t = jnp.ones(1) * t

        embed = jnp.squeeze(self.act(self.embed_dense(self.embed(t))))

        dx = x
        for i, (layer, embed_dense) in enumerate(zip(self.layers, self.embed_denses)):
            dx = layer(dx) + embed_dense(embed)
            if i < len(self.layers) - 1:
                dx = self.act(dx)

        return dx

class DenseNet3(nn.Module):
    dim: int  # the ambient dimension of the problem
    key: jnp.ndarray
    hidden_dims = [64, 64, 64]
    embed_dim: int = 64


    def setup(self):
        self.embed = GaussianFourierProjection(embed_dim=self.embed_dim, key=self.key)
        self.embed_dense = nn.Dense(features=self.embed_dim)
        self.layers = [nn.Dense(dim_out) for dim_out in self.hidden_dims + [self.dim]]
        self.embed_denses = [nn.Dense(dim_out) for dim_out in self.hidden_dims + [self.dim]]
        # for dim_out in self.hidden_dims + [self.dim]:
        #     layer = ConcatSquashLinear(dim_out)
        #     self.layers.append(layer)
        self.act = lambda x: x * jax.nn.sigmoid(x)


    def __call__(self, t, x):


        if type(t) is float:
            t = jnp.ones(1) * t
        elif t.ndim == 0:
            t = jnp.ones(1) * t

        embed = jnp.squeeze(self.act(self.embed_dense(self.embed(t))))

        dx = x
        for i, (layer, embed_dense) in enumerate(zip(self.layers, self.embed_denses)):
            dx = layer(dx) + embed_dense(embed)
            if i < len(self.layers) - 1:
                dx = self.act(dx)

        return jnp.squeeze(dx)