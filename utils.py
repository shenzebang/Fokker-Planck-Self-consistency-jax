import jax.numpy as jnp


def marginal_prob_std(t: float, sigma: float) -> float:
    return jnp.sqrt((sigma ** (2*t) - 1.)/2./jnp.log(sigma))


def diffusion_coeff(t: float, sigma: float) -> float:
    return sigma**t


