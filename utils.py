import jax.numpy as jnp
# import objax
import jax

v_matmul = jax.vmap(jnp.matmul, in_axes=(None, 0))

def marginal_prob_std(t: float, sigma: float) -> float:
    return jnp.sqrt((sigma ** (2 * t) - 1.) / 2. / jnp.log(sigma))


def diffusion_coeff(t: float, sigma: float) -> float:
    return sigma ** t


# def flatten(vars: objax.VarCollection):
#     return jnp.concatenate([jnp.ravel(param) for param in vars], axis=0)


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
        _tensors.append(flat_tensor[prev_ind:prev_ind + flat_size].reshape(shape))
        prev_ind += flat_size

    return _tensors


def _divergence_fn(f, _x, _v):
    # Hutchinson’s Estimator
    # computes the divergence of net at x with random vector v
    _, u = jax.jvp(f, (_x,), (_v,))
    # print(u.shape, _x.shape, _v.shape)
    return jnp.sum(u * _v)


# f_list = [lambda x: f(x)[i]]

def _divergence_bf_fn(f, _x):
    # brute-force implementation of the divergence operator
    # _x should be a d-dimensional vector
    jacobian = jax.jacfwd(f)
    a = jacobian(_x)
    return jnp.sum(jnp.diag(a))



batch_div_bf_fn = jax.vmap(_divergence_bf_fn, in_axes=[None, 0])

batch_div_fn = jax.vmap(_divergence_fn, in_axes=[None, None, 0])


def divergence_fn(f, _x, _v=None):
    if _v is None:
        return batch_div_bf_fn(f, _x)
    else:
        return batch_div_fn(f, _x, _v).mean(axis=0)


