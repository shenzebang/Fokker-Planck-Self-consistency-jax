import objax
import jax.numpy as jnp
import jax
import objax.nn as nn


class ConcatSquashLinear(objax.Module):
    def __init__(self, dim_in: int, dim_out: int):
        self._layer = objax.nn.Linear(dim_in, dim_out)
        self._hyper_bias = objax.nn.Linear(1, dim_out, use_bias=False)
        self._hyper_gate = objax.nn.Linear(1, dim_out)

    def __call__(self, t: jnp.ndarray, x: jnp.DeviceArray):
        # print(x.shape)
        # print(t.shape)
        return self._layer(x) * jax.nn.sigmoid(self._hyper_gate(t.reshape(-1, 1))) \
            + self._hyper_bias(t.reshape(-1, 1))

class ScoreNet(objax.Module):
    def __init__(self, hdim: int):
        self._layer1 = ConcatSquashLinear(28*28, hdim)
        self._layer2 = ConcatSquashLinear(hdim, hdim)
        self._layer3 = ConcatSquashLinear(hdim, 28*28)

    def __call__(self, t, x):
        x = self._layer1(t, x)
        x = self._layer2(t, x)
        x = self._layer3(t, x)
        # print(x.shape)
        return x


class GaussianFourierProjection(objax.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, key, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = jax.random.normal(key, [embed_dim // 2])
        self.W = self.W * scale

    def __call__(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

class Dense(objax.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def __call__(self, x):
        return self.dense(x)[..., None, None]

class UNet(objax.Module):
    """A time-dependent score-based model built upon U-Net architecture."""
    def __init__(self, marginal_prob_std, key, channels=jnp.array([32, 64, 128, 256]), embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential([GaussianFourierProjection(embed_dim=embed_dim, key=key),
                                   nn.Linear(embed_dim, embed_dim)])
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2D(1, channels[0], 3, strides=1, use_bias=False, padding=1)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.BatchNorm2D(nin=channels[0])
        self.conv2 = nn.Conv2D(channels[0], channels[1], 3, strides=2, use_bias=False, padding=1)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.BatchNorm2D(nin=channels[1])
        self.conv3 = nn.Conv2D(channels[1], channels[2], 3, strides=2, use_bias=False, padding=1)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.BatchNorm2D(nin=channels[2])
        self.conv4 = nn.Conv2D(channels[2], channels[3], 3, strides=2, use_bias=False, padding=0)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.BatchNorm2D(nin=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2D(channels[3], channels[2], 3, strides=2, use_bias=False, padding='VALID')
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.BatchNorm2D(nin=channels[2])
        self.tconv3 = nn.ConvTranspose2D(channels[2] + channels[2], channels[1], 3, strides=2, use_bias=False,
                                         padding='SAME')
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.BatchNorm2D(nin=channels[1])
        self.tconv2 = nn.ConvTranspose2D(channels[1] + channels[1], channels[0], 3, strides=2, use_bias=False,
                                         padding='SAME')
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.BatchNorm2D(nin=channels[0])
        self.tconv1 = nn.ConvTranspose2D(channels[0] + channels[0], 1, 3, strides=1)

        # The swish activation function
        self.act = lambda x: x * jax.nn.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def __call__(self, t, x, training):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # print(embed.shape)
        # print(x.shape)
        # print(t.shape)
        # assert not torch.isnan(torch.sum(embed))
        # Encoding path
        h1 = self.conv1(x)
        # print(h1.shape)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        # print("shape", self.dense1(embed).shape)
        ## Group normalization
        h1 = self.gnorm1(h1, training=training)
        h1 = self.act(h1)
        # print(h1.shape)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2, training=training)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3, training=training)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4, training=training)
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
        h += self.dense5(embed)
        h = self.tgnorm4(h, training=training)
        h = self.act(h)
        h = self.tconv3(jnp.concatenate([h, h3], axis=1))
        # print(h.shape)

        h += self.dense6(embed)
        h = self.tgnorm3(h, training=training)
        h = self.act(h)
        h = self.tconv2(jnp.concatenate([h, h2], axis=1))
        # print(h.shape)

        h += self.dense7(embed)
        h = self.tgnorm2(h, training=training)
        h = self.act(h)
        # print(h.shape)

        h = self.tconv1(jnp.concatenate([h, h1], axis=1))
        # print(h.shape)

        # assert not torch.isnan(torch.sum(h))

        # Normalize output
        # h = h / self.marginal_prob_std(t)[:, None, None, None]
        # print(self.marginal_prob_std(t))
        # assert not torch.isnan(torch.sum(h))
        return h