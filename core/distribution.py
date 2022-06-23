import jax.numpy as jnp
import jax.random
import jax.random as random
from utils import v_matmul
from typing import List

class Distribution(object):
    def sample(self, batch_size: int, key):
        raise NotImplementedError

    def score(self, x: jnp.ndarray):
        raise NotImplementedError

    def logdensity(self, x: jnp.ndarray):
        raise NotImplementedError


class Gaussian(Distribution):
    def __init__(self, mu: jnp.ndarray, sigma: jnp.ndarray):
        self.dim = mu.shape[0]
        self.mu = mu
        self.sigma = sigma


        if sigma.shape[0] != 1:
            assert sigma.shape[0] == sigma.shape[1] # make sure sigma is a square matrix
            # if sigma.shape[0] != 1, the covariance matrix is sigma.transpose * sigma
            self.cov = jnp.matmul(self.sigma, jnp.transpose(self.sigma))
            self.inv_cov = jnp.linalg.inv(self.cov)
            self.log_det = jnp.log(jnp.linalg.det(self.cov * 2 * jnp.pi))
        else:
            # if sigma.shape[0] == 1, the covariance matrix is (sigma ** 2) * jnp.eye(mu.shape[0])
            self.cov = self.sigma ** 2
            self.inv_cov = 1./self.cov
            self.log_det = None # this is not used

    def sample(self, batch_size: int, key):
        if self.sigma.shape[0] == 1:
            return self.sigma * random.normal(key, (batch_size, self.dim)) + self.mu
        else:
            return v_matmul(self.sigma, random.normal(key, (batch_size, self.dim))) + self.mu

    def score(self, x: jnp.ndarray):
        if self.sigma.shape[0] == 1:
            return (self.mu - x) / self.sigma**2
        else:
            return v_matmul(self.inv_cov, self.mu - x)


    def logdensity(self, x: jnp.ndarray):
        if self.sigma.shape[0] == 1:
            return -self.dim/2 * jnp.log(2*jnp.pi*self.sigma ** 2) - jnp.sum((x - self.mu)**2, axis=(1,)) / 2/self.sigma**2
        else:
            quad = jnp.dot(x - self.mu, jnp.matmul(self.inv_cov, x - self.mu))
            return - .5 * (self.log_det + quad)

class GaussianMixture(Distribution):
    def __init__(self, mus: List[jnp.ndarray], sigmas: List[jnp.ndarray]):
        # we assume uniform weight among the Gaussians
        self.n_Gaussians = len(mus)
        assert self.n_Gaussians == len(sigmas)
        # assert self.n_Gaussians == weights.shape[0]
        # assert all(weights > 0)

        # self.weights = weights / jnp.sum(weights)
        self.dim = mus[0].shape[0]


        self.mus = mus
        self.sigmas = sigmas
        self.covs, self.inv_covs, self.dets = [], [], []

        for sigma in sigmas:
            if sigma.ndim != 0:
                assert sigma.shape[0] == sigma.shape[1] # make sure sigma is a square matrix
                # if sigma.shape[0] != 1, the covariance matrix is sigma.transpose * sigma
                cov = jnp.matmul(sigma, jnp.transpose(sigma))
                inv_cov = jnp.linalg.inv(cov)
                det = jnp.linalg.det(cov)
            else:
                # sigma is a scalar
                cov = sigma ** 2
                inv_cov = 1./cov
                det = sigma ** (2 * self.dim)

            self.covs.append(cov)
            self.inv_covs.append(inv_cov)
            self.dets.append(det)

        self.mus = jnp.stack(self.mus)
        self.covs = jnp.stack(self.covs)
        self.inv_covs = jnp.stack(self.inv_covs)
        self.dets = jnp.stack(self.dets)

    def sample(self, batch_size: int, key):
        n_sample_per_center = []
        remainder = batch_size % self.n_Gaussians
        for i in range(self.n_Gaussians):
            n_sample_i = batch_size // self.n_Gaussians
            if remainder != 0:
                n_sample_i += 1
                remainder -= 1
            n_sample_per_center.append(n_sample_i)

        samples = []
        keys = jax.random.split(key, self.n_Gaussians)
        for i, (n_sample_i, _key) in enumerate(zip(n_sample_per_center, keys)):
            mu, sigma = self.mus[i, :], self.sigmas[i]
            if sigma.ndim == 0:
                samples.append(sigma * random.normal(_key, (n_sample_i, self.dim)) + mu)
            else:
                samples.append(v_matmul(sigma, random.normal(_key, (n_sample_i, self.dim))) + mu)

        return jnp.concatenate(samples, axis=0)

    def logdensity(self, xs: jnp.ndarray):
        return v_logdensity_gmm(xs, self.mus, self.inv_covs, self.dets)

    def score(self, xs: jnp.ndarray):
        return v_score_gmm(xs, self.mus, self.inv_covs, self.dets)

def _density_gaussian(x, mu, inv_cov, det):
    # computes the density in a single Gaussian of a single point
    a = x - mu
    dim = x.shape[0]
    if inv_cov.ndim == 0:
        return jnp.exp(- .5 * jnp.dot(a, a) * inv_cov) / jnp.sqrt((2 * jnp.pi) ** dim * det)
    else:
        return jnp.exp(- .5 * jnp.dot(a, jnp.matmul(inv_cov, a))) / jnp.sqrt((2*jnp.pi) ** dim * det)

v_density_gaussian = jax.vmap(_density_gaussian, in_axes=[None, 0, 0, 0])
# computes the density in several Gaussians of a single point


def _logdensity_gmm(x, mus, inv_covs, dets):
    # computes log densities of gmm of multiple points
    densities = v_density_gaussian(x, mus, inv_covs, dets)
    # densities : (self.n_Gaussians)
    return jnp.log(jnp.mean(densities, axis=0))

v_logdensity_gmm = jax.vmap(_logdensity_gmm, in_axes=[0, None, None, None])
# computes log densities of gmm of multiple points

_score_gmm = jax.grad(_logdensity_gmm)
# compute the gradient w.r.t. x

v_score_gmm = jax.vmap(_score_gmm, in_axes=[0, None, None, None])
