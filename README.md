# Self-Consistency-JAX

This repo contains code accompaning the following papers (and more to come)
1. [Self-Consistency of the Fokker-Planck Equation (Shen et al, COLT 2022)](https://arxiv.org/abs/2206.00860)


## Results
We consider an example where the initial distribution is Gaussian, i.e. $\alpha_0 = \mathcal{N}(\mu_0, \Sigma_0)$, and the drifting term is a quadratic function, i.e. $V(x) = (x - \mu_\infty)^\top\Sigma_\infty^{-1}(x - \mu_\infty)$.
We use this example since we know the analytical solution of the FPE $\alpha(t, x)$ in this specific instance and hence we can explicitly calculate the difference between the learned hypothesis velocity field $f_\theta$ and the ground truth.
Specifically, we know that for any time $t \geq 0$, 
the solution $\alpha(t, \cdot) = \mathcal{N}(\mu_t, \Gamma_t^\top\Gamma_t)$ 
is a Gaussian distribution where $\mu_t$ 
and $\Gamma_t$ evolve in the following manner 

$$\frac{\mathrm{d} \mu_t}{\mathrm{d} t} = \Sigma_\infty^{-1}(\mu_\infty - \mu_t),\quad \frac{\mathrm{d} \Gamma_t}{\mathrm{d} t} = -\Sigma^{-1}_\infty \Gamma_t + {\Gamma_t^{-1}}^\top, \Gamma_0 = \sqrt{\Sigma_0}, $$

if we take the the domain $\mathcal{X} = \mathbb{R}^2$. 
In our experiment, we take $\mu_0 = (-4, -4)$,
$\Sigma_0 = \mathrm{diag}(0.7, 1.3)$, 
and $\mu_\infty = (4, 4)$, 
$\Sigma_\infty = \mathrm{diag}(1.1, 0.9)$.
## Usage

To replicate the results presented in our submission, please use the `*.sh` files under the `script` folder.

## dependence


