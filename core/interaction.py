import jax
import jax.numpy as jnp

class InteractionKernel(object):
    def conv_grad(self, x: jnp.DeviceArray):
        raise NotImplementedError


class LogRepulsion_QuadAttraction(InteractionKernel):
    def conv_grad(self, x: jnp.DeviceArray):
        pairwise_diff = x[:, None, :] - x[None, :, :]
        pairwise_distance = jnp.sum(pairwise_diff ** 2, axis=(2,)) + 1e-10*jnp.eye(x.shape[0])
        res_0 = pairwise_diff
        # res_1 = jnp.nan_to_num(pairwise_diff / pairwise_distance[:, :, None])
        res_1 = pairwise_diff / pairwise_distance[:, :, None]
        return jnp.sum(res_0 - res_1, axis=(1,)) / (x.shape[0] - 1)


class NavierStokesKernel2D(InteractionKernel):
    def conv_grad(self, x: jnp.DeviceArray):
        assert x.shape[1] == 2
        pairwise_diff = x[:, None, :] - x[None, :, :]

        # add a negligible term to the diagonal so that "0/0=0"
        pairwise_distance = jnp.sum(pairwise_diff ** 2, axis=(2,)) + 1e-10 * jnp.eye(x.shape[0])
        res = jnp.stack([-pairwise_diff[:, :, 1], pairwise_diff[:, :, 0]], axis=2) / pairwise_distance[:, :, None]
        return - jnp.sum(res, axis=1) / (x.shape[0] - 1) / 2 / jnp.pi