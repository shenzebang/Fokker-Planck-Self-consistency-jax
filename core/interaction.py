import jax
import jax.numpy as jnp

class InteractionKernel(object):
    def conv_grad(self, x:jnp.ndarray, y:jnp.ndarray):
        raise NotImplementedError


class LogRepulsion_QuadAttraction(InteractionKernel):
    def conv_grad(self, x:jnp.ndarray, y:jnp.ndarray):
        pairwise_diff = x[:, None, :] - y[None, :, :]
        pairwise_distance = jnp.maximum(jnp.sum(pairwise_diff ** 2, axis=(2,)), 1e-10)
        res_0 = pairwise_diff
        # res_1 = jnp.nan_to_num(pairwise_diff / pairwise_distance[:, :, None])
        res_1 = pairwise_diff / pairwise_distance[:, :, None]
        return jnp.sum(res_0 - res_1, axis=(1,)) / y.shape[0]


class NavierStokesKernel2D(InteractionKernel):
    def conv_grad(self, x: jnp.ndarray, y: jnp.ndarray):
        # x are the points where the conv_grad are evaluated
        # y are the reference points to approximate the convolution
        assert x.shape[1] == 2 and y.shape[1] == 2
        pairwise_diff = x[:, None, :] - y[None, :, :]

        # add a negligible term so that "0/0=0"
        pairwise_distance = jnp.maximum(jnp.sum(pairwise_diff ** 2, axis=(2,)), 1e-10)
        res = jnp.stack([-pairwise_diff[:, :, 1], pairwise_diff[:, :, 0]], axis=2) / pairwise_distance[:, :, None]
        return - jnp.sum(res, axis=1) / y.shape[0] / 2 / jnp.pi