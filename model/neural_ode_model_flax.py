import jax.numpy as jnp
import jax
from flax import linen as nn


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
