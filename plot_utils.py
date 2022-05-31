import jax.numpy as jnp
from linear_drifting_diffusion import scale, tolerance, T
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
from utils import divergence_fn
import numpy as np
from PIL import Image
import jax

batch_size_plot = 1000


def plot_result(net, params, init_distribution, target_potential, T):
    bar_f = lambda _x, _t, _params: net.apply(_params, _t, _x) * scale - target_potential.gradient(_x)
    # compute x(T) by solve IVP (I) & compute the actor loss
    # ================ Forward ===================
    x_0 = init_distribution.sample(batch_size_plot)
    states_0 = [x_0]

    def ode_func1(states, t):
        x = states[0]
        bar_f_t_theta = lambda _x: bar_f(_x, t, params)
        dx = bar_f_t_theta(x)

        return [dx]

    tspace = jnp.array((0., T))
    result_forward = odeint(ode_func1, states_0, tspace, atol=tolerance, rtol=tolerance)
    x_T = result_forward[0][1]
    # ================ Forward ===================

    print(jnp.mean(x_T, axis=(0,)))
    # print(x_T.shape)
    plt.scatter(x_T[:, 0], x_T[:, 1])
    plt.savefig('Gaussian_to_Gaussian.png')
    plt.show()


def plt_flow_2d(prior_logdensity, target_potential, net, params, end_T=T, n_frames=20, npts=256, LOW=-6, HIGH=6):
    # bar_f = lambda _x, _t: net.apply(params, _t, _x) - target_potential.gradient(_x)
    time_per_frame = np.true_divide(end_T, n_frames)
    imgs = []
    for i in range(n_frames):
        end_T_i = time_per_frame * (i+1)
        pTx_T = plt_density_2d_jit(prior_logdensity, target_potential, net, params, end_T_i, npts, LOW, HIGH, True)
        imgs.append(Image.fromarray(np.array(pTx_T/pTx_T.max()) * 255))
    imgs[0].save('nwgf.gif', save_all=True, append_images=imgs[1:], duration=40)



def plt_density_2d(prior_logdensity, bar_f, end_T=T, npts=256, LOW=-6, HIGH=6,
                   gif_subroutine=False):
    side = jnp.linspace(LOW, HIGH, npts)
    xx, yy = jnp.meshgrid(side, side)
    x = jnp.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    states_T = [x]

    def ode_func1(states, t):
        t = end_T - t
        x = states[0]
        dx = bar_f(x, t)
        return [-dx]

    tspace = jnp.array((0., end_T))
    result_backward = odeint(ode_func1, states_T, tspace, atol=tolerance, rtol=tolerance)
    x_0 = result_backward[0][1]

    log_p0x_0 = prior_logdensity(x_0)
    states_0 = [x_0, log_p0x_0]

    def ode_func2(states, t):
        x = states[0]
        dx = bar_f(x, t)

        bar_f_t = lambda _x: bar_f(_x, t)
        div_bar_f_t = lambda _x: divergence_fn(bar_f_t, _x)
        dlog_ptx_t = - div_bar_f_t(x)
        return [dx, dlog_ptx_t]

    tspace = jnp.array((0., end_T))
    result_forward = odeint(ode_func2, states_0, tspace, atol=tolerance, rtol=tolerance)
    x_T = result_forward[0][1]
    log_pTx_T = result_forward[1][1]


    pTx_T = jnp.exp(log_pTx_T).reshape(npts, npts)

    if not gif_subroutine:
        print("numerical error %.5f" % (jnp.mean(jnp.sum((x_T - x) ** 2, axis=(1,)))))
        # ax = plt.gca()
        plt.imshow(pTx_T)
        plt.savefig('Gaussian_to_Gaussian.png')
        plt.show()

        print(f"The total mass between [{LOW, HIGH}]^2 is {((HIGH - LOW) / npts) ** 2 * jnp.sum(pTx_T)}")
    else:
        return pTx_T

def _plt_density_2d(prior_logdensity, target_potential, net, params, end_T=T, npts=256, LOW=-6, HIGH=6,
                   gif_subroutine=False):
    bar_f = lambda _x, _t: net.apply(params, _t, _x) - target_potential.gradient(_x)
    side = jnp.linspace(LOW, HIGH, npts)
    xx, yy = jnp.meshgrid(side, side)
    x = jnp.hstack([xx.reshape(-1, 1), jnp.flip(yy.reshape(-1, 1))])
    states_T = [x]

    def ode_func1(states, t):
        t = end_T - t
        x = states[0]
        dx = bar_f(x, t)
        return [-dx]

    tspace = jnp.array((0., end_T))
    result_backward = odeint(ode_func1, states_T, tspace, atol=tolerance, rtol=tolerance)
    x_0 = result_backward[0][1]

    log_p0x_0 = prior_logdensity(x_0)
    states_0 = [x_0, log_p0x_0]

    def ode_func2(states, t):
        x = states[0]
        dx = bar_f(x, t)

        bar_f_t = lambda _x: bar_f(_x, t)
        div_bar_f_t = lambda _x: divergence_fn(bar_f_t, _x)
        dlog_ptx_t = - div_bar_f_t(x)
        return [dx, dlog_ptx_t]

    tspace = jnp.array((0., end_T))
    result_forward = odeint(ode_func2, states_0, tspace, atol=tolerance, rtol=tolerance)
    x_T = result_forward[0][1]
    log_pTx_T = result_forward[1][1]


    pTx_T = jnp.exp(log_pTx_T).reshape(npts, npts)

    if not gif_subroutine:
        print("numerical error %.5f" % (jnp.mean(jnp.sum((x_T - x) ** 2, axis=(1,)))))
        # ax = plt.gca()
        plt.imshow(pTx_T)
        plt.savefig('Gaussian_to_Gaussian.png')
        plt.show()

        print(f"The total mass between [{LOW, HIGH}]^2 is {((HIGH - LOW) / npts) ** 2 * jnp.sum(pTx_T)}")
    else:
        return pTx_T

    # Compute the total mass in the region


plt_density_2d_jit = jax.jit(_plt_density_2d, static_argnums=(0, 1, 2, 4, 5, 6, 7, 8))







def plt_density_1d(prior_logdensity, bar_f, npts=256, LOW=-6, HIGH=6):
    x = jnp.linspace(LOW, HIGH, npts)[:, None]

    states_T = [x]

    def ode_func1(states, t):
        t = T - t
        x = states[0]
        dx = bar_f(x, t)
        return [-dx]

    tspace = jnp.array((0., T))
    result_backward = odeint(ode_func1, states_T, tspace, atol=tolerance, rtol=tolerance)
    x_0 = result_backward[0][1]

    log_p0x_0 = prior_logdensity(x_0)
    states_0 = [x_0, log_p0x_0]

    def ode_func2(states, t):
        x = states[0]
        dx = bar_f(x, t)

        bar_f_t = lambda _x: bar_f(_x, t)
        div_bar_f_t = lambda _x: divergence_fn(bar_f_t, _x)
        dlog_ptx_t = - div_bar_f_t(x)
        return [dx, dlog_ptx_t]

    tspace = jnp.array((0., T))
    result_forward = odeint(ode_func2, states_0, tspace, atol=tolerance, rtol=tolerance)
    x_T = result_forward[0][1]
    log_pTx_T = result_forward[1][1]

    pTx_T = jnp.exp(log_pTx_T)
    print("numerical error %.5f" % (jnp.mean(jnp.sum((x_T - x) ** 2, axis=(1,)))))
    # ax = plt.gca()
    # plt.imshow(pTx_T)
    plt.plot(x, pTx_T)
    plt.savefig('Gaussian_to_Gaussian.png')
    plt.show()
