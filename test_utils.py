import jax.numpy as jnp
import jax
from jax.experimental.ode import odeint
from flax import linen as nn

v_matmul = jax.vmap(jnp.matmul, in_axes=(None, 0))

def _gaussian_score(x, Sigma, mu): # return the score for a given Gaussian(mu, Sigma) at x
    cov = jnp.matmul(Sigma, jnp.transpose(Sigma))
    inv_cov = jax.numpy.linalg.inv(cov)
    return jnp.matmul(inv_cov, mu - x)

# return the score for a given Gaussians(Sigma, mu) at [x1, ..., xN]
v_gaussian_score = jax.vmap(_gaussian_score, in_axes=[0, None, None])



def eval_Gaussian_score_FP(Sigma_0, mu_0, Sigma, mu, time_stamps, grid_points, beta, tolerance = 1e-5):
    states_0 = [Sigma_0, mu_0]
    cov = jnp.matmul(Sigma, jnp.transpose(Sigma))
    inv_cov = jax.numpy.linalg.inv(cov)
    def ode_func(states, t):
        Sigma_t, mu_t = states
        dSigma = - jnp.matmul(inv_cov, Sigma_t) + beta * jnp.transpose(jax.numpy.linalg.inv(Sigma_t))
        dmu = jnp.matmul(inv_cov, mu - mu_t)
        return [dSigma, dmu]

    v_gaussian_score_grid_points = lambda _Sigma, _mu: v_gaussian_score(grid_points, _Sigma, _mu)
    v_v_gaussian_score_grid_points = jax.vmap(v_gaussian_score_grid_points, in_axes=[0, 0])

    states = odeint(ode_func, states_0, time_stamps, atol=tolerance, rtol=tolerance)

    scores = v_v_gaussian_score_grid_points(states[0], states[1])

    return scores


def test_Gaussian_score(net: nn.Module, params, time_stamps, grid_points, Gaussian_score):

    v_net_apply = jax.vmap(net.apply, in_axes=[None, 0, None])
    negative_scores_pred = v_net_apply(params, time_stamps, grid_points)

    return jnp.mean(jnp.sum((negative_scores_pred + Gaussian_score) ** 2, axis=(2,)))