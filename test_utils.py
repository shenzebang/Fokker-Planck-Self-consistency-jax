import jax.numpy as jnp
import jax
from jax.experimental.ode import odeint
from flax import linen as nn
from utils import divergence_fn, v_matmul



domain_size = 5.

def _gaussian_score(x, Sigma, mu): # return the score for a given Gaussian(mu, Sigma) at x
    cov = jnp.matmul(Sigma, jnp.transpose(Sigma))
    inv_cov = jax.numpy.linalg.inv(cov)
    return jnp.matmul(inv_cov, mu - x)

# return the score for a given Gaussians(Sigma, mu) at [x1, ..., xN]
v_gaussian_score = jax.vmap(_gaussian_score, in_axes=[0, None, None])


def _gaussian_log_density(x, Sigma, mu):
    cov = jnp.matmul(Sigma, jnp.transpose(Sigma))
    log_det = jnp.log(jax.numpy.linalg.det(cov * 2 * jnp.pi))
    inv_cov = jax.numpy.linalg.inv(cov)
    quad = jnp.dot(x - mu, jnp.matmul(inv_cov, x - mu))
    return - .5 * (log_det + quad)

v_gaussian_log_density = jax.vmap(_gaussian_log_density, in_axes=[0, None, None])

def _gaussian_sample(n_samples, Sigma, mu, key):
    return v_matmul(Sigma, jax.random.normal(key, (n_samples, Sigma.shape[0]))) + mu

v_gaussian_sample = jax.vmap(_gaussian_sample, in_axes=[None, 0, 0, 0])


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





def gaussian_sample_score_log_density_FP(Sigma_0, mu_0, Sigma, mu, time_stamps, n_samples, beta, key, tolerance = 1e-5):
    states_0 = [Sigma_0, mu_0]
    cov = jnp.matmul(Sigma, jnp.transpose(Sigma))
    inv_cov = jax.numpy.linalg.inv(cov)
    def ode_func(states, t):
        Sigma_t, mu_t = states
        dSigma = - jnp.matmul(inv_cov, Sigma_t) + beta * jnp.transpose(jax.numpy.linalg.inv(Sigma_t))
        dmu = jnp.matmul(inv_cov, mu - mu_t)
        return [dSigma, dmu]
    states = odeint(ode_func, states_0, time_stamps, atol=tolerance, rtol=tolerance)
    # sample
    keys = jax.random.split(key, time_stamps.shape[0])
    positions = v_gaussian_sample(n_samples, states[0], states[1], keys)

    # score
    v_v_gaussian_score = jax.vmap(v_gaussian_score, in_axes=[0, 0, 0])
    scores = v_v_gaussian_score(positions, states[0], states[1])

    # log-density
    v_v_gaussian_log_density = jax.vmap(v_gaussian_log_density, in_axes=[0, 0, 0])
    log_densities = v_v_gaussian_log_density(positions, states[0], states[1])

    return positions, scores, log_densities

def eval_Gaussian_score_and_log_density_FP(Sigma_0, mu_0, Sigma, mu, time_stamps, grid_points, beta, tolerance = 1e-5):
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

    v_gaussian_log_density_grid_points = lambda _Sigma, _mu: v_gaussian_log_density(grid_points, _Sigma, _mu)
    v_v_gaussian_log_density_grid_points = jax.vmap(v_gaussian_log_density_grid_points, in_axes=[0, 0])


    states = odeint(ode_func, states_0, time_stamps, atol=tolerance, rtol=tolerance)

    scores = v_v_gaussian_score_grid_points(states[0], states[1])
    log_densities = v_v_gaussian_log_density_grid_points(states[0], states[1])

    return scores, log_densities


def _log_density_2d_grid(params, velocity, prior_logdensity, end_T, npts=101, LOW=-domain_size, HIGH=domain_size):
    tolerance = 1e-5
    x, y = jnp.linspace(LOW, HIGH, num=npts), jnp.linspace(LOW, HIGH, num=npts)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([jnp.reshape(xx, (-1)), jnp.reshape(yy, (-1))], axis=1)

    _velocity = lambda _x, _t: velocity(params, _x, _t)
    states_T = [grid_points]
    def ode_func1(states, t):
        t = end_T - t
        x = states[0]
        dx = _velocity(x, t)
        return [-dx]

    tspace = jnp.array((0., end_T))
    result_backward = odeint(ode_func1, states_T, tspace, atol=tolerance, rtol=tolerance)
    x_0 = result_backward[0][1]


    log_p0x_0 = prior_logdensity(x_0)
    states_0 = [x_0, log_p0x_0]

    def ode_func2(states, t):
        x = states[0]
        dx = _velocity(x, t)

        bar_f_t = lambda _x: _velocity(_x, t)
        div_bar_f_t = lambda _x: divergence_fn(bar_f_t, _x)
        dlog_ptx_t = - div_bar_f_t(x)
        return [dx, dlog_ptx_t]

    tspace = jnp.array((0., end_T))
    result_forward = odeint(ode_func2, states_0, tspace, atol=tolerance, rtol=tolerance)
    x_T = result_forward[0][1]
    log_pTx_T = result_forward[1][1]

    return log_pTx_T

# log_density = jax.vmap(_log_density_2d_grid, in_axes=[None, None, None, 0])

def _log_density(params, velocity, prior_logdensity, end_T, positions):
    tolerance = 1e-5
    _velocity = lambda _x, _t: velocity(params, _x, _t)
    states_T = [positions]
    def ode_func1(states, t):
        t = end_T - t
        x = states[0]
        dx = _velocity(x, t)
        return [-dx]

    tspace = jnp.array((0., end_T))
    result_backward = odeint(ode_func1, states_T, tspace, atol=tolerance, rtol=tolerance)
    x_0 = result_backward[0][1]


    log_p0x_0 = prior_logdensity(x_0)
    states_0 = [x_0, log_p0x_0]

    def ode_func2(states, t):
        x = states[0]
        dx = _velocity(x, t)

        bar_f_t = lambda _x: _velocity(_x, t)
        div_bar_f_t = lambda _x: divergence_fn(bar_f_t, _x)
        dlog_ptx_t = - div_bar_f_t(x)
        return [dx, dlog_ptx_t]

    tspace = jnp.array((0., end_T))
    result_forward = odeint(ode_func2, states_0, tspace, atol=tolerance, rtol=tolerance)
    x_T = result_forward[0][1]
    log_pTx_T = result_forward[1][1]

    return log_pTx_T

log_density = jax.vmap(_log_density, in_axes=[None, None, None, 0, None])
