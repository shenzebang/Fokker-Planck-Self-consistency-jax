import jax.numpy as jnp
from core.distribution import Distribution
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
from utils import divergence_fn
import numpy as np
from PIL import Image
import jax
import seaborn as sns
from matplotlib import animation
from typing import List

batch_size_plot = 1000


# def plot_result(net, params, init_distribution, target_potential, T):
#     bar_f = lambda _x, _t, _params: net.apply(_params, _t, _x)  - target_potential.gradient(_x)
#     # compute x(T) by solve IVP (I) & compute the actor loss
#     # ================ Forward ===================
#     x_0 = init_distribution.sample(batch_size_plot)
#     states_0 = [x_0]
#
#     def ode_func1(states, t):
#         x = states[0]
#         bar_f_t_theta = lambda _x: bar_f(_x, t, params)
#         dx = bar_f_t_theta(x)
#
#         return [dx]
#
#     tspace = jnp.array((0., T))
#     result_forward = odeint(ode_func1, states_0, tspace, atol=tolerance, rtol=tolerance)
#     x_T = result_forward[0][1]
#     # ================ Forward ===================
#
#     print(jnp.mean(x_T, axis=(0,)))
#     # print(x_T.shape)
#     plt.scatter(x_T[:, 0], x_T[:, 1])
#     plt.savefig('Gaussian_to_Gaussian.png')
#     plt.show()





# def plt_density_2d(prior_logdensity, bar_f, end_T=T, npts=256, LOW=-6, HIGH=6,
#                    gif_subroutine=False):
#     side = jnp.linspace(LOW, HIGH, npts)
#     xx, yy = jnp.meshgrid(side, side)
#     x = jnp.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
#     states_T = [x]
#
#     def ode_func1(states, t):
#         t = end_T - t
#         x = states[0]
#         dx = bar_f(x, t)
#         return [-dx]
#
#     tspace = jnp.array((0., end_T))
#     result_backward = odeint(ode_func1, states_T, tspace, atol=tolerance, rtol=tolerance)
#     x_0 = result_backward[0][1]
#
#     log_p0x_0 = prior_logdensity(x_0)
#     states_0 = [x_0, log_p0x_0]
#
#     def ode_func2(states, t):
#         x = states[0]
#         dx = bar_f(x, t)
#
#         bar_f_t = lambda _x: bar_f(_x, t)
#         div_bar_f_t = lambda _x: divergence_fn(bar_f_t, _x)
#         dlog_ptx_t = - div_bar_f_t(x)
#         return [dx, dlog_ptx_t]
#
#     tspace = jnp.array((0., end_T))
#     result_forward = odeint(ode_func2, states_0, tspace, atol=tolerance, rtol=tolerance)
#     x_T = result_forward[0][1]
#     log_pTx_T = result_forward[1][1]
#
#
#     pTx_T = jnp.exp(log_pTx_T).reshape(npts, npts)
#
#     if not gif_subroutine:
#         print("numerical error %.5f" % (jnp.mean(jnp.sum((x_T - x) ** 2, axis=(1,)))))
#         # ax = plt.gca()
#         plt.imshow(pTx_T)
#         plt.savefig('Gaussian_to_Gaussian.png')
#         plt.show()
#
#         print(f"The total mass between [{LOW, HIGH}]^2 is {((HIGH - LOW) / npts) ** 2 * jnp.sum(pTx_T)}")
#     else:
#         return pTx_T
#
# def _plt_density_2d(prior_logdensity, target_potential, net, params, end_T=T, npts=256, LOW=-6, HIGH=6,
#                    gif_subroutine=False):
#     bar_f = lambda _x, _t: net.apply(params, _t, _x) - target_potential.gradient(_x)
#     side = jnp.linspace(LOW, HIGH, npts)
#     xx, yy = jnp.meshgrid(side, side)
#     x = jnp.hstack([xx.reshape(-1, 1), jnp.flip(yy.reshape(-1, 1))])
#     states_T = [x]
#
#     def ode_func1(states, t):
#         t = end_T - t
#         x = states[0]
#         dx = bar_f(x, t)
#         return [-dx]
#
#     tspace = jnp.array((0., end_T))
#     result_backward = odeint(ode_func1, states_T, tspace, atol=tolerance, rtol=tolerance)
#     x_0 = result_backward[0][1]
#
#     log_p0x_0 = prior_logdensity(x_0)
#     states_0 = [x_0, log_p0x_0]
#
#     def ode_func2(states, t):
#         x = states[0]
#         dx = bar_f(x, t)
#
#         bar_f_t = lambda _x: bar_f(_x, t)
#         div_bar_f_t = lambda _x: divergence_fn(bar_f_t, _x)
#         dlog_ptx_t = - div_bar_f_t(x)
#         return [dx, dlog_ptx_t]
#
#     tspace = jnp.array((0., end_T))
#     result_forward = odeint(ode_func2, states_0, tspace, atol=tolerance, rtol=tolerance)
#     x_T = result_forward[0][1]
#     log_pTx_T = result_forward[1][1]
#
#
#     pTx_T = jnp.exp(log_pTx_T).reshape(npts, npts)
#
#     if not gif_subroutine:
#         print("numerical error %.5f" % (jnp.mean(jnp.sum((x_T - x) ** 2, axis=(1,)))))
#         # ax = plt.gca()
#         plt.imshow(pTx_T)
#         plt.savefig('Gaussian_to_Gaussian.png')
#         plt.show()
#
#         print(f"The total mass between [{LOW, HIGH}]^2 is {((HIGH - LOW) / npts) ** 2 * jnp.sum(pTx_T)}")
#     else:
#         return pTx_T

    # Compute the total mass in the region


# plt_density_2d_jit = jax.jit(_plt_density_2d, static_argnums=(0, 1, 2, 4, 5, 6, 7, 8))







# def plt_density_1d(prior_logdensity, bar_f, npts=256, LOW=-6, HIGH=6):
#     x = jnp.linspace(LOW, HIGH, npts)[:, None]
#
#     states_T = [x]
#
#     def ode_func1(states, t):
#         t = T - t
#         x = states[0]
#         dx = bar_f(x, t)
#         return [-dx]
#
#     tspace = jnp.array((0., T))
#     result_backward = odeint(ode_func1, states_T, tspace, atol=tolerance, rtol=tolerance)
#     x_0 = result_backward[0][1]
#
#     log_p0x_0 = prior_logdensity(x_0)
#     states_0 = [x_0, log_p0x_0]
#
#     def ode_func2(states, t):
#         x = states[0]
#         dx = bar_f(x, t)
#
#         bar_f_t = lambda _x: bar_f(_x, t)
#         div_bar_f_t = lambda _x: divergence_fn(bar_f_t, _x)
#         dlog_ptx_t = - div_bar_f_t(x)
#         return [dx, dlog_ptx_t]
#
#     tspace = jnp.array((0., T))
#     result_forward = odeint(ode_func2, states_0, tspace, atol=tolerance, rtol=tolerance)
#     x_T = result_forward[0][1]
#     log_pTx_T = result_forward[1][1]
#
#     pTx_T = jnp.exp(log_pTx_T)
#     print("numerical error %.5f" % (jnp.mean(jnp.sum((x_T - x) ** 2, axis=(1,)))))
#     # ax = plt.gca()
#     # plt.imshow(pTx_T)
#     plt.plot(x, pTx_T)
#     plt.savefig('Gaussian_to_Gaussian.png')
#     plt.show()


# def plt_density_2d_new(initial_distribution: Distribution, bar_f, n_samples = 10000, end_T=T, n_frames = 100, LOW=-6, HIGH=6):
#     data = initial_distribution.sample(n_samples)
#     states_0 = [data]
#
#     def ode_func1(states, t):
#         x = states[0]
#         dx = bar_f(x, t)
#         return [dx]
#
#     tspace = jnp.linspace(0, end_T, n_frames)
#
#     result_forward = odeint(ode_func1, states_0, tspace, atol=tolerance, rtol=tolerance)
#
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     def animate(num):
#         state = result_forward[0][num]
#         x, y = state[:, 0], state[:, 1]
#         ax.clear()
#         sns.scatterplot(x=x, y=y, s=5, color=".15")
#         sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
#         sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
#         ax.set_xlim(-10, 10)
#         ax.set_ylim(-10, 10)
#
#     anim = animation.FuncAnimation(fig, animate, frames=len(result_forward[0]), blit=False)
#     fig.tight_layout()
#     anim.save('plot/contour.gif', writer='imagemagick', fps=5)
#     plt.show()


def plot_velocity_field_2d(args, f_velocity, interval=50):

    x, y = jnp.linspace(-args.plot_domain_size, args.plot_domain_size, num=41), jnp.linspace(-args.plot_domain_size, args.plot_domain_size, num=41)
    xx, yy = jnp.meshgrid(x, y)
    grid_points = jnp.stack([jnp.reshape(xx, (-1)), jnp.reshape(yy, (-1))], axis=1)


    velocity_0 = f_velocity(grid_points, 0.)

    fig, ax = plt.subplots()
    Q = ax.quiver(grid_points[:, 0], grid_points[:, 1], velocity_0[:, 0], velocity_0[:, 1], pivot='mid', color='r', units='inches')
    ax.set_xlim(jnp.min(grid_points[:, 0]), jnp.max(grid_points[:, 0]))
    ax.set_ylim(jnp.min(grid_points[:, 1]), jnp.max(grid_points[:, 1]))

    frames = int(args.total_evolving_time / interval * 1000)
    def update_quiver(num, Q):
        velocity = f_velocity(grid_points, num * interval / 1000.)
        Q.set_UVC(velocity[:, 0], velocity[:, 1])
        return Q

    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q,), frames=frames, interval=interval, blit=False)
    fig.tight_layout()
    file_name = f"{args.plot_save_directory}/{args.PDE}/{args.total_evolving_time}_{args.diffusion_coefficient}_velocity.gif"
    anim.save(file_name, writer='imagemagick', fps=2)
    # plt.show()
    plt.close(fig)

def plot_density_contour_2d(args, density_data: List[jnp.ndarray]):
    fig, ax = plt.subplots(figsize=(6, 6))
    pal = sns.dark_palette("navy", as_cmap=True)

    def animate(num):
        state = density_data[num]
        x, y = state[:, 0], state[:, 1]
        ax.clear()
        # sns.scatterplot(x=x, y=y, s=5, color=".15")
        # sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
        sns.kdeplot(x=x, y=y, levels=10, color="w", linewidths=1, cmap=pal)
        ax.set_xlim(-args.plot_domain_size, args.plot_domain_size)
        ax.set_ylim(-args.plot_domain_size, args.plot_domain_size)

    anim = animation.FuncAnimation(fig, animate, frames=len(density_data), blit=False)
    fig.tight_layout()
    file_name = f"{args.plot_save_directory}/{args.PDE}/{args.total_evolving_time}_{args.diffusion_coefficient}_density_contour.gif"
    anim.save(file_name, writer='imagemagick', fps=2)
    # plt.show()
    plt.close(fig)

def plot_trajectory_2d(args, trajectories: List[jnp.ndarray], plot_multiple=1.1):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['red', 'green', 'blue', 'yellow', 'magenta']
    # plot the trajectory
    for i, trajectory in enumerate(trajectories):
        assert trajectory.shape[1] == 2  # the data should be 2D
        plt.quiver(trajectory[:-1, 0], trajectory[:-1, 1],
                   trajectory[1:, 0]-trajectory[:-1, 0], trajectory[1:, 1]-trajectory[:-1, 1],
                   scale_units='xy', angles='xy', scale=1, color=colors[i % len(colors)])

    # mark the start and end points of every trajectory with scatter
    ## gather the start and end points
    start_points = jnp.stack([trajectory[0, :] for trajectory in trajectories], axis=0)
    plt.scatter(start_points[:, 0], start_points[:, 1], marker='o', linewidths=.5, color='b')
    end_points = jnp.stack([trajectory[-1, :] for trajectory in trajectories], axis=0)
    plt.scatter(end_points[:, 0], end_points[:, 1], marker='v', linewidths=.5, color='b')

    # particle may leave the domain of plot. increase the x/y limit to include them in the plot
    ax.set_xlim(-args.plot_domain_size * plot_multiple, args.plot_domain_size * plot_multiple)
    ax.set_ylim(-args.plot_domain_size * plot_multiple, args.plot_domain_size * plot_multiple)

    fig.tight_layout()
    file_name = f"{args.plot_save_directory}/{args.PDE}/{args.total_evolving_time}_{args.diffusion_coefficient}_trajecotry.png"
    plt.savefig(file_name, dpi=600)
    plt.close(fig)