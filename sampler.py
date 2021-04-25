from jax.experimental.ode import odeint
import jax.numpy as jnp
import jax.random as random
from jax import jit

import numpy as np
error_tolerance = 1e-5  # @param {'type': 'number'}

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                key,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                eps=1e-3):

    t_1 = jnp.ones([batch_size,])
    init_x = random.normal(key, (batch_size, 1, 28, 28)) * marginal_prob_std(t_1)[:, None, None, None]
    def ode_func(x, t):
        t = 1. - t
        time_steps = jnp.ones((x.shape[0],)) * t
        g = diffusion_coeff(t)
        dx = -.5 * (g**2) * score_model(time_steps, x, training=False)
        return -dx

    ode_func = jit(ode_func)
    tspace = np.array((0., 1.))
    res = odeint(ode_func, init_x, tspace, atol=atol, rtol=rtol)[1]
    return res.reshape(res.shape[0], 1, 28, 28)
