import jax.numpy as jnp
import objax


def marginal_prob_std(t: float, sigma: float) -> float:
    return jnp.sqrt((sigma ** (2*t) - 1.)/2./jnp.log(sigma))


def diffusion_coeff(t: float, sigma: float) -> float:
    return sigma**t

def flatten(vars: objax.VarCollection):
    return jnp.concatenate([jnp.ravel(param) for param in vars], axis=0)

def reshape_like(flat_tensor, shape_list):
    prev_ind = 0
    _tensors = []
    for shape in shape_list:
        # print(var.shape)
        flat_size = jnp.prod(jnp.asarray(shape))
        # print(shape)
        # print(shape[0])
        # flat_size = int(jnp.prod(shape[0]))
        # flat_size = int(shape)
        _tensors.append(flat_tensor[prev_ind:prev_ind+flat_size].reshape(shape))
        prev_ind += flat_size

    return _tensors




