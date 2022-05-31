import jax.numpy as jnp
import jax.random as random
from utils import v_matmul

class Distribution(object):
    def sample(self, batch_size: int):
        raise NotImplementedError

    def score(self, x: jnp.ndarray):
        raise NotImplementedError

    def logdensity(self, x: jnp.ndarray):
        raise NotImplementedError


class Gaussian(Distribution):
    def __init__(self, mu: jnp.ndarray, sigma: jnp.ndarray, key):
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
        self.key = key

    def sample(self, batch_size):
        self.key, subkey = random.split(self.key)
        if self.sigma.shape[0] == 1:
            return self.sigma * random.normal(subkey, (batch_size, self.dim)) + self.mu
        else:
            return v_matmul(self.sigma, random.normal(subkey, (batch_size, self.dim))) + self.mu

    def score(self, x):
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