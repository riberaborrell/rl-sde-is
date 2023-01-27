import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm, rc, rcParams
from shapely.geometry import Polygon

from rl_sde_is.utils_figures import TITLES_FIG, COLORS_FIG, COLORS_TAB20b

def get_state_action_1d_extent(env):
    ''' set extent bounds for 1d state space in the x-axis and
        1d action space in the y-axis
    '''

    extent = env.state_space_h[0], env.state_space_h[-1], \
             env.action_space_h[0], env.action_space_h[-1]
    return extent

def get_state_2d_extent(env):
    ''' set extent bounds for 2d state space
    '''

    extent = env.state_space_h[0, 0, 0], env.state_space_h[-1, -1, 0], \
             env.state_space_h[0, 0, 1], env.state_space_h[-1, -1, 1]
    return extent

def plot_episode_states_1d(env, ep_states, loc=None):

    # compute potential at each state
    ep_pot = env.potential(ep_states)

    # draw target set
    target_set = Polygon([
       (1, 0),
       (1, 15),
       (3, 15),
       (3, 0),
    ])
    target_set_x, target_set_y = target_set.exterior.xy

    # compute potential on the grid
    potential = env.potential(env.state_space_h)

    fig, ax = plt.subplots()
    ax.set_title(r'Trajectory')
    ax.set_xlabel(r'States')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 4)
    ax.text(1.12, 2.2, r'Target set', size=13, rotation=0.)
    ax.fill(target_set_x, target_set_y, alpha=0.4, fc='tab:orange', ec='none')
    ax.plot(env.state_space_h, potential, label=r'Potential $V_\alpha$')
    ax.scatter(ep_states[::1], ep_pot[::1], alpha=.1, color='black', marker='o', s=100)
    plt.legend(loc=loc)
    plt.show()

def plot_episode_states_2d(env, ep_states):

    # draw target set
    target_set = Polygon([
       (1, 1),
       (1, 2),
       (2, 2),
       (2, 1),
    ])
    target_set_x, target_set_y = target_set.exterior.xy

    # compute potential on the grid
    flat_state_space_h = env.state_space_h.reshape(env.n_states, env.d)
    potential = env.potential(flat_state_space_h).reshape(env.state_space_h.shape[:-1])

    fig, ax = plt.subplots()
    ax.set_title(r'Trajectory')
    ax.set_xlabel(r'$s_1$')
    ax.set_xlabel(r'$s_2$')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    #ax.text(1.12, 2.2, r'Target set', size=13, rotation=0.)
    ax.fill(target_set_x, target_set_y, alpha=0.4, fc='tab:orange', ec='none')
    ax.imshow(
        potential,
        origin='lower',
        extent=get_state_2d_extent(env),
        cmap=cm.coolwarm,
    )
    ax.scatter(ep_states[:, 0], ep_states[:, 1], alpha=.1, color='black', marker='o', s=100)
    plt.show()


def plot_returns_episodes(returns, run_mean_returns, loc=None):
    fig, ax = plt.subplots()
    ax.set_title('Returns')
    ax.set_xlabel('Episodes')
    ax.set_ylim(-10, 0)

    plt.plot(returns, label='return')
    plt.plot(run_mean_returns, label='running mean of last returns')
    plt.legend(loc=loc)
    plt.show()

def plot_run_var_returns_episodes(run_var_returns):
    fig, ax = plt.subplots()
    ax.set_title('Running variance Returns')
    ax.set_xlabel('Episodes')

    plt.semilogy(run_var_returns)
    #plt.legend()
    plt.show()

def plot_run_mean_returns_with_error_episodes(run_mean_returns, run_var_returns):
    fig, ax = plt.subplots()
    ax.set_title('Returns')
    ax.set_xlabel('Episodes')
    ax.set_ylim(-10, 0)

    n_episodes = run_mean_returns.shape[0]
    x = np.arange(n_episodes)
    y = run_mean_returns
    error = np.sqrt(run_var_returns)
    ax.plot(x, y, label='running mean of last returns')
    ax.fill_between(x, y-error, y+error, alpha=0.5, label='standard deviation')
    plt.legend()
    plt.show()

def plot_time_steps_episodes(time_steps, avg_time_steps):
    fig, ax = plt.subplots()
    ax.set_title('Time steps')
    ax.set_xlabel('Episodes')

    plt.plot(time_steps, label='time steps')
    plt.plot(avg_time_steps, label='running mean of last time steps')
    plt.legend()
    plt.show()


def plot_expected_returns_epochs(test_mean_returns):
    fig, ax = plt.subplots()
    ax.set_title('Expected return')
    ax.set_xlabel('Epochs')
    ax.set_ylim(-10, 0)

    plt.plot(test_mean_returns)
    #plt.legend()
    plt.show()

def plot_var_returns_epochs(test_var_returns):
    fig, ax = plt.subplots()
    ax.set_title('Sample variance return')
    ax.set_xlabel('Epochs')

    plt.semilogy(test_var_returns)
    #plt.legend()
    plt.show()

def plot_expected_returns_with_error_epochs(test_mean_returns, test_var_returns):
    fig, ax = plt.subplots()
    ax.set_title('Expected Return')
    ax.set_xlabel('Epochs')
    ax.set_ylim(-10, 0)

    n_epochs = test_mean_returns.shape[0]
    x = np.arange(n_epochs)
    y = test_mean_returns
    error = np.sqrt(test_var_returns)
    ax.plot(x, y, label='expected return')
    ax.fill_between(x, y-error, y+error, alpha=0.5, label='standard deviation')
    plt.legend()
    plt.show()

def plot_loss_epochs(losses):
    fig, ax = plt.subplots()
    ax.set_title('Loss function')
    ax.set_xlabel('Epochs') #ax.set_ylim(-10, 0)
    plt.plot(losses)
    #plt.legend()
    plt.show()

def plot_var_losses_epochs(var_losses):
    fig, ax = plt.subplots()
    ax.set_title('Sample variance loss')
    ax.set_xlabel('Epochs')

    plt.semilogy(test_var_returns)
    #plt.legend()
    plt.show()

def plot_losses_with_errors_epochs(losses, var_losses):
    fig, ax = plt.subplots()
    ax.set_title('Estimated Loss function')
    ax.set_xlabel('Epochs')
    #ax.set_ylim(-10, 0)
    n_epochs = len(losses)
    x = np.arange(n_epochs)
    y = losses
    error = np.sqrt(var_losses)
    ax.plot(x, y, label='estimated loss')
    ax.fill_between(x, y-error, y+error, alpha=0.5, label='standard deviation')
    plt.legend()
    plt.show()

def plot_time_steps_epochs(time_steps):
    fig, ax = plt.subplots()
    ax.set_title('Estimated time steps')
    ax.set_xlabel('Epochs')
    plt.plot(time_steps)
    #plt.legend()
    plt.show()

def plot_det_policy_l2_error_epochs(l2_errors, ylim=None):
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy-l2-error'])
    ax.set_xlabel('Epochs')
    #ax.set_xlim(0, l2_errors.shape[0])
    ax.semilogy(l2_errors)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.show()

def plot_det_policy_l2_error_ct_epochs(cts, l2_errors, ylim=None):
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy-l2-error'])
    ax.set_xlabel('CT(s)')
    #ax.set_xlim(0, l2_errors.shape[0])
    ax.semilogy(cts, l2_errors)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.show()

def plot_det_policy_l2_error_iterations(l2_errors, iterations=None, ylim=None):
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy-l2-error'])
    ax.set_xlabel('Gradient steps')
    #ax.set_xlim(0, l2_errors.shape[0])
    if iterations is not None:
        plt.semilogy(iterations, l2_errors)
    else:
        plt.semilogy(l2_errors)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.show()

def plot_det_policy_l2_error_episodes(l2_errors, episodes=None, ylim=None):
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy-l2-error'])
    ax.set_xlabel('Episodes')
    #ax.set_xlim(0, l2_errors.shape[0])
    if episodes is not None:
        plt.semilogy(episodes, l2_errors)
    else:
        plt.semilogy(l2_errors)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.show()

def plot_reward_table(env, r_table):

    fig, ax = plt.subplots()
    ax.set_title('Reward Table')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = ax.imshow(
        r_table.T,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.coolwarm,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_frequency(env, n_table):

    fig, ax = plt.subplots()
    ax.set_title('Frequency')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = ax.imshow(
        n_table.T,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.coolwarm,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_q_value_function_1d(env, q_table):

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['q-value-function'])
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = fig.axes[0].imshow(
        q_table.T,
        vmax=0,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.viridis,
        aspect='auto',
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_advantage_function_1d(env, a_table, policy_critic=None, vmin=None):

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['a-value-function'])
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = fig.axes[0].imshow(
        a_table.T,
        vmin=vmin,
        vmax=0,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.plasma,
        aspect='auto',
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if policy_critic is not None:
        x = env.state_space_h
        ax.plot(x, policy_critic, c='grey', linestyle='--')

    plt.show()

def plot_value_function_1d(env, value_function, value_function_hjb, loc=None):

    fig, ax = plt.subplots()
    ax.set_title('Value function')
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])

    plt.plot(env.state_space_h, value_function)
    plt.plot(env.state_space_h, -value_function_hjb, label=r'hjb', linestyle=':')
    plt.legend(loc=loc)
    plt.show()

def plot_value_function_2d(env, value_function, value_function_hjb):

    fig, ax = plt.subplots()
    ax.set_title(r'Value function $V(s) = max_a Q_\omega(s, a)$')
    ax.set_xlabel(r'$s_1$')
    ax.set_ylabel(r'$s_2$')

    im = fig.axes[0].imshow(
        value_function.T,
        vmin=np.min(value_function_hjb),
        vmax=np.max(value_function_hjb),
        origin='lower',
        extent=get_state_2d_extent(env),
        cmap=cm.coolwarm,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()


def plot_value_function_1d_actor_critic(env, value_function_critic_initial, value_function_critic,
                                        value_function_actor_critic, value_function_hjb,
                                        ylim=None, loc=None):
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['value-function'])
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])
    if ylim is not None:
        ax.set_ylim(ylim)

    x = env.state_space_h
    ax.plot(x, value_function_critic_initial, label=r'initial critic', c='tab:blue')
    ax.plot(x, value_function_critic, label=r'critic: $V_\omega(s) = max_a Q_\omega(s, a)$',
            c='tab:orange')
    ax.plot(x, value_function_actor_critic,
             label=r'actor-critic: $ V_{\theta, \omega}(s) = Q_\omega(s, \mu_\theta(s))$',
            c='tab:purple')
    ax.plot(x, -value_function_hjb, label=r'hjb', color=COLORS_FIG['hjb'], linestyle=':')
    ax.legend(loc=loc)
    plt.show()

def plot_stoch_policy(env, action_prob_dists, policy, control_hjb, loc=None):

    # plot action probability distributions
    fig, ax = plt.subplots()
    ax.set_title('Action Probability distributions')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')
    im = ax.imshow(
        action_prob_dists.T,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.plasma,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

    # plot sampled actions following policy
    fig, ax = plt.subplots()
    ax.set_title('Stochastic Policy')
    ax.set_xlabel('States')
    ax.set_ylabel('Sampled actions')
    plt.scatter(env.state_space_h, policy)
    plt.plot(env.state_space_h, control_hjb[:, 0], label=r'hjb solution')
    plt.legend(loc=loc)
    plt.show()

def plot_mu_and_simga_gaussian_stoch_policy(env, mu, sigma_sq):

    # plot mu
    fig, ax = plt.subplots()
    ax.plot(env.state_space_h, mu)
    ax.set_title('mu')
    ax.set_xlabel('State space')
    plt.show()

    # plot sigma
    fig, ax = plt.subplots()
    ax.plot(env.state_space_h, sigma_sq)
    ax.set_title('sigma')
    ax.set_xlabel('State space')
    plt.show()

    # plot mu with error
    fig, ax = plt.subplots()
    x = env.state_space_h
    y = mu
    error = np.sqrt(sigma_sq)
    ax.plot(x, y, label='mu')
    ax.fill_between(x, y-error, y+error, alpha=0.5, label='standard deviation')
    ax.set_title('mu')
    ax.set_xlabel('State space')
    plt.show()

def plot_det_policy_1d(env, policy, control_hjb, loc=None):

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])

    ax.plot(env.state_space_h, policy)
    ax.plot(env.state_space_h, control_hjb[:, 0], label=r'hjb', color=COLORS_FIG['hjb'], linestyle=':')
    ax.legend(loc=loc)
    plt.show()

def plot_det_policies_1d(env, policies, control_hjb, labels=None, colors=None,
                         ylim=None, loc='upper right'):

    n_policies = policies.shape[0]

    if labels is None:
        labels = ['' for i in range(n_policies + 1)]

    if colors is None:
        colors = [None for i in range(n_policies + 1)]

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])
    if ylim is not None:
        ax.set_ylim(ylim)

    x = env.state_space_h
    for i in range(n_policies):
        ax.plot(x, policies[i], color=colors[i], label=labels[i])
    ax.plot(x, control_hjb, color=colors[i+1], linestyle=':', label=labels[i+1])

    if labels[0]:
        plt.legend(loc=loc)
    plt.show()

def plot_det_policies_1d_black_and_white(env, policies, control_hjb):
    n_policies = policies.shape[0]
    cmap = cm.get_cmap('Greys')
    colors = cmap(np.linspace(0, 1, n_policies))

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])

    for i in range(n_policies):
        ax.plot(env.state_space_h, policies[i], c=colors[i])
    ax.plot(env.state_space_h, control_hjb, c='black', ls='-.')
    #ax.set_ylim(-3, 3)
    plt.show()

def plot_det_policy_1d_actor_critic(env, policy_actor_initial, policy_critic_initial, policy_actor,
                                    policy_critic, control_hjb, ylim=None, loc=None):
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])
    if ylim is not None:
        ax.set_ylim(ylim)

    x = env.state_space_h
    ax.plot(x, policy_actor_initial, label=r'initial actor', c='tab:blue')
    #ax.plot(x, policy_critic_initial, label=r'initial actor', c='tab:red')
    ax.plot(x, policy_actor, label=r'actor: $\mu_\theta(s)$', c='tab:green')
    ax.plot(x, policy_critic, label=r'critic: $\mu_\omega(s) = argmax_a Q_\omega(s, a)$',
            c='tab:orange')
    ax.plot(x, control_hjb[:, 0], label=r'hjb', color='black', linestyle=':')
    ax.legend(loc=loc)
    plt.show()

def plot_det_policy_2d(env, policy, policy_hjb):
    X = env.state_space_h[:, :, 0]
    Y = env.state_space_h[:, :, 1]
    U = policy[:, :, 0]
    V = policy[:, :, 1]
    X, Y, U, V = coarse_quiver_arrows(U, V, X, Y, l=25)

    U_hjb = policy_hjb[:, :, 0]
    V_hjb = policy_hjb[:, :, 1]
    _, _, U_hjb, V_hjb = coarse_quiver_arrows(U_hjb, V_hjb, l=25)

    # initialize figure
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlabel(r'$s_1$')
    ax.set_ylabel(r'$s_2$')
    ax.set_xlim(env.state_space_low, env.state_space_high)
    ax.set_ylim(env.state_space_low, env.state_space_high)

    # initialize norm object
    C = np.sqrt(U**2 + V**2)
    C_hjb = np.sqrt(U_hjb**2 + V_hjb**2)
    norm = colors.Normalize(vmin=np.min(C_hjb), vmax=np.max(C_hjb))

    # vector field plot
    Q = ax.quiver(
        X,
        Y,
        U,
        V,
        C,
        norm=norm,
        cmap=cm.viridis,
        angles='xy',
        scale_units='xy',
        width=0.005,
    )

    # add space for color bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(Q, cax=cbar_ax)

    plt.show()

def initialize_det_policy_1d_figure(env, policy, control_hjb):

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))

    # turn interactive mode on
    plt.ion()

    ax.set_title(TITLES_FIG['policy'], fontsize=10)
    ax.set_xlabel('States', fontsize=8)
    ax.set_xlim(env.state_space_low, env.state_space_high)
    ax.set_ylim(env.action_space_low, env.action_space_high)
    det_policy_line = ax.plot(env.state_space_h, policy)[0]
    ax.plot(env.state_space_h, control_hjb, label=r'hjb', color=COLORS_FIG['hjb'])

    plt.legend()
    plt.show()

    return det_policy_line

def update_det_policy_1d_figure(env, policy, line):
    line.set_data(env.state_space_h, policy)
    plt.pause(0.1)

def coarse_quiver_arrows(U, V, X=None, Y=None, l=25):
    kx = U.shape[0] // 25
    ky = U.shape[1] // 25
    if X is not None:
        X = X[::kx, ::ky]
    if Y is not None:
        Y = Y[::kx, ::ky]
    U = U[::kx, ::ky]
    V = V[::kx, ::ky]
    return X, Y, U, V

def initialize_det_policy_2d_figure(env, policy, policy_hjb):
    X = env.state_space_h[:, :, 0]
    Y = env.state_space_h[:, :, 1]
    U = policy[:, :, 0]
    V = policy[:, :, 1]
    X, Y, U, V = coarse_quiver_arrows(U, V, X, Y, l=25)

    U_hjb = policy_hjb[:, :, 0]
    V_hjb = policy_hjb[:, :, 1]
    _, _, U_hjb, V_hjb = coarse_quiver_arrows(U_hjb, V_hjb, l=25)

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title(TITLES_FIG['policy'], fontsize=10)
    ax.set_xlabel(r'$s_1$', fontsize=8)
    ax.set_ylabel(r'$s_2$', fontsize=8)
    ax.set_xlim(env.state_space_low, env.state_space_high)
    ax.set_ylim(env.state_space_low, env.state_space_high)

    # turn interactive mode on
    plt.ion()

    # initialize norm object and make rgba array
    C = np.sqrt(U**2 + V**2)
    C_hjb = np.sqrt(U_hjb**2 + V_hjb**2)
    norm = colors.Normalize(vmin=np.min(C_hjb), vmax=np.max(C_hjb))

    Q_policy = ax.quiver(
        X,
        Y,
        U,
        V,
        C,
        norm=norm,
        cmap=cm.viridis,
        angles='xy',
        scale_units='xy',
        width=0.005,
    )

    # add space for color bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(Q_policy, cax=cbar_ax)

    plt.show()

    return Q_policy

def update_det_policy_2d_figure(env, policy, Q_policy):

    U = policy[:, :, 0]
    V = policy[:, :, 1]
    _, _, U, V = coarse_quiver_arrows(U, V, l=25)
    C = np.sqrt(U**2 + V**2)

    # update plots
    Q_policy.set_UVC(U, V, C)

    # update figure frequency
    plt.pause(0.1)

def canvas_det_policy_2d_figure(env, data, backup_episodes, control_hjb):
    import torch
    from rl_sde_is.td3_core import load_backup_models
    from rl_sde_is.approximate_methods import compute_det_policy_actions

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title(TITLES_FIG['policy'], fontsize=10)
    ax.set_xlabel(r'$s_1$', fontsize=8)
    ax.set_ylabel(r'$s_2$', fontsize=8)
    ax.set_xlim(env.state_space_low, env.state_space_high)
    ax.set_ylim(env.state_space_low, env.state_space_high)

    # turn interactive mode on
    plt.ion()

    X = env.state_space_h[:, :, 0]
    Y = env.state_space_h[:, :, 1]

    U_hjb = control_hjb[:, :, 0]
    V_hjb = control_hjb[:, :, 1]
    X, Y, U_hjb, V_hjb = coarse_quiver_arrows(U_hjb, V_hjb, X, Y, l=25)

    # initialize norm object and make rgba array
    C_hjb = np.sqrt(U_hjb**2 + V_hjb**2)
    norm = colors.Normalize(vmin=np.min(C_hjb), vmax=np.max(C_hjb))

    Q_policy = ax.quiver(
        X,
        Y,
        U_hjb,
        V_hjb,
        C_hjb,
        norm=norm,
        cmap=cm.viridis,
        angles='xy',
        scale_units='xy',
        width=0.005,
    )
    plt.show()

    # get models
    actor = data['actor']
    critic = data['critic1']

    # looop to update figures
    for ep in backup_episodes:

        # load policy
        load_backup_models(data, ep)
        states = torch.FloatTensor(env.state_space_h)
        policy = compute_det_policy_actions(env, actor, states)
        U = policy[:, :, 0]
        V = policy[:, :, 1]
        _, _, U, V = coarse_quiver_arrows(U, V, l=25)
        C = np.sqrt(U**2 + V**2)

        # update plots
        Q_policy.set_UVC(U, V, C)

        # update figure frequency
        plt.pause(0.1)


def initialize_return_and_time_steps_figures(env, n_episodes):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 4))
    ax1, ax2 = axes
    fig.tight_layout(pad=2.5)

    # turn interactive mode on
    plt.ion()

    # returns
    ax1.set_title('Return', fontsize=10)
    ax1.set_xlabel('Episodes', fontsize=8)
    ax1.set_xlim(0, n_episodes)
    ax1.set_ylim(-20, 1)

    ax2.set_title('Time steps', fontsize=10)
    ax2.set_xlabel('Episodes', fontsize=8)
    ax2.set_xlim(0, n_episodes)
    ax2.set_ylim(0, 5000)

    # episodes and array with nan values
    episodes = np.arange(n_episodes)
    nan_array = np.empty_like(episodes, dtype=np.float32)
    nan_array.fill(np.nan)

    # time steps
    line_returns = ax1.plot(episodes, nan_array)[0]
    line_avg_returns = ax1.plot(episodes, nan_array)[0]

    # time steps
    line_time_steps = ax2.plot(episodes, nan_array)[0]
    line_avg_time_steps = ax2.plot(episodes, nan_array)[0]

    return (line_returns, line_avg_returns, line_time_steps, line_avg_time_steps)

def update_return_and_time_steps_figures(env, returns, time_steps, lines):

    # unpack lines and images
    line_returns, line_avg_returns, line_time_steps, line_avg_time_steps = lines

    # episodes
    n_episodes = returns.shape[0]
    episodes = np.arange(n_episodes)

    # update plots
    line_returns.set_data(episodes, returns)
    #line_avg_returns.set_data(episodes, avg_returns)
    line_time_steps.set_data(episodes, time_steps)
    #line_avg_time_steps.set_data(episodes, avg_time_steps)

    # update figure frequency
    plt.pause(0.1)

def initialize_q_learning_figures(env, q_table, v_table, a_table, policy, value_function_hjb, control_hjb):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]

    # turn interactive mode on
    plt.ion()

    # q table
    ax1.set_title(TITLES_FIG['q-value-function'])
    ax1.set_xlabel('States')
    ax1.set_ylabel('Actions')
    im_q_table = ax1.imshow(
        q_table.T,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.viridis,
        #aspect='auto',
    )

    # value function
    ax2.set_title('Value function')
    ax2.set_xlabel('States')
    line_value_function = ax2.plot(env.state_space_h, v_table)[0]
    ax2.plot(env.state_space_h, -value_function_hjb)

    # a table
    ax3.set_title(TITLES_FIG['a-value-function'])
    ax3.set_xlabel('States')
    ax3.set_ylabel('Actions')
    im_a_table = ax3.imshow(
        a_table.T,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.plasma,
        #aspect='auto',
    )

    # control
    ax4.set_title(TITLES_FIG['policy'])
    ax4.set_xlabel('States')
    ax4.set_ylabel('Actions')
    line_control = ax4.plot(env.state_space_h, policy)[0]
    ax4.plot(env.state_space_h, control_hjb)

    plt.show()
    return (im_q_table, line_value_function, im_a_table, line_control)

def update_q_learning_figures(env, q_table, v_table, a_table, policy, tuples):

    # unpack lines and images
    im_q_table, line_value_function, im_a_table, line_policy = tuples

    # update plots
    im_q_table.set_data(q_table.T)
    line_value_function.set_data(env.state_space_h, v_table)
    im_a_table.set_data(a_table.T)
    line_policy.set_data(env.state_space_h, policy)

    # update figure frequency
    plt.pause(0.1)

def initialize_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic, a_table,
                                    policy_actor, policy_critic, value_function_hjb, control_hjb):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 4))
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]
    fig.tight_layout(pad=2.5)

    # turn interactive mode on
    plt.ion()

    # q table
    ax1.set_title(TITLES_FIG['q-value-function'], fontsize=10)
    ax1.set_xlabel('States', fontsize=8)
    ax1.set_ylabel('Actions', fontsize=8)
    ax1.set_xlim(env.state_space_low, env.state_space_high)
    ax1.set_ylim(env.action_space_low, env.action_space_high)
    im_q_table = ax1.imshow(
        q_table.T,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.viridis,
        aspect='auto',
    )

    # value function
    ax2.set_title('Value function', fontsize=10)
    ax2.set_xlabel('States', fontsize=8)
    ax2.set_xlim(env.state_space_low, env.state_space_high)
    line_value_f_actor_critic = ax2.plot(env.state_space_h, v_table_actor_critic)[0]
    line_value_f_critic = ax2.plot(env.state_space_h, v_table_critic)[0]
    ax2.plot(env.state_space_h, -value_function_hjb)

    # a table
    ax3.set_title(TITLES_FIG['a-value-function'], fontsize=10)
    ax3.set_xlabel('States', fontsize=8)
    ax3.set_ylabel('Actions', fontsize=8)
    ax3.set_xlim(env.state_space_low, env.state_space_high)
    ax3.set_ylim(env.action_space_low, env.action_space_high)
    im_a_table = ax3.imshow(
        a_table.T,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.plasma,
        aspect='auto',
    )

    # control
    ax4.set_title(TITLES_FIG['policy'], fontsize=10)
    ax4.set_xlabel('States', fontsize=8)
    ax4.set_ylabel('Actions', fontsize=8)
    ax4.set_xlim(env.state_space_low, env.state_space_high)
    ax4.set_ylim(env.action_space_low, env.action_space_high)
    line_policy_actor = ax4.plot(env.state_space_h, policy_actor)[0]
    line_policy_critic = ax4.plot(env.state_space_h, policy_critic)[0]
    ax4.plot(env.state_space_h, control_hjb)

    plt.show()
    return (im_q_table, line_value_f_actor_critic, line_value_f_critic, im_a_table,
            line_policy_actor, line_policy_critic)

def update_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic, a_table,
                                policy_actor, policy_critic, tuples):

    # unpack lines and images
    im_q_table, line_value_f_actor_critic, line_value_f_critic, im_a_table, \
            line_policy_actor, line_policy_critic = tuples

    # update plots
    im_q_table.set_data(q_table.T)
    line_value_f_actor_critic.set_data(env.state_space_h, v_table_actor_critic)
    line_value_f_critic.set_data(env.state_space_h, v_table_critic)
    im_a_table.set_data(a_table.T)
    line_policy_actor.set_data(env.state_space_h, policy_actor)
    line_policy_critic.set_data(env.state_space_h, policy_critic)

    # update figure frequency
    plt.pause(0.1)

def canvas_actor_critic_1d_figures(env, data, backup_episodes, value_function_hjb, control_hjb):
    from rl_sde_is.td3_core import load_backup_models
    from rl_sde_is.approximate_methods import compute_tables_critic, compute_tables_actor_critic

    # get models
    actor = data['actor']
    critic = data['critic1']

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]
    plt.show()
    fig.canvas.draw()
    fig.tight_layout(pad=4.0)

    # q table
    ax1.set_title(TITLES_FIG['q-value-function'])
    ax1.set_xlabel('States')
    ax1.set_ylabel('Actions')
    ax1.set_xlim(env.state_space_low, env.state_space_high)
    ax1.set_ylim(env.action_space_low, env.action_space_high)

    # a table
    ax3.set_title(TITLES_FIG['a-value-function'])
    ax3.set_xlabel('States')
    ax3.set_ylabel('Actions')
    ax3.set_xlim(env.state_space_low, env.state_space_high)
    ax3.set_ylim(env.action_space_low, env.action_space_high)

    # looop to update figures
    for ep in backup_episodes:

        # load models
        load_backup_models(data, ep)
        q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic)
        v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)

        # update title
        fig.suptitle('episode: {:d}'.format(ep), fontsize='x-large')

        # q table
        ax1.imshow(
            q_table.T,
            origin='lower',
            extent=get_state_action_1d_extent(env),
            cmap=cm.viridis,
            aspect='auto',
        )

        # value function
        ax2.cla()
        ax2.set_title('Value function')
        ax2.set_xlabel('States')
        ax2.set_xlim(env.state_space_low, env.state_space_high)
        ax2.plot(env.state_space_h, v_table_actor_critic)
        ax2.plot(env.state_space_h, v_table_critic)
        ax2.plot(env.state_space_h, -value_function_hjb)

        # a table
        ax3.imshow(
            a_table.T,
            origin='lower',
            extent=get_state_action_1d_extent(env),
            cmap=cm.plasma,
            aspect='auto',
        )

        ax4.cla()
        ax4.set_title(TITLES_FIG['policy'])
        ax4.set_xlabel('States')
        ax4.set_ylabel('Actions')
        ax4.set_xlim(env.state_space_low, env.state_space_high)
        ax4.set_ylim(env.action_space_low, env.action_space_high)
        ax4.plot(env.state_space_h, policy_actor)
        ax4.plot(env.state_space_h, policy_critic)
        ax4.plot(env.state_space_h, control_hjb[:, 0], color='black', linestyle=':')

        # pause
        plt.pause(0.01)

        # draw
        fig.canvas.draw()

def plot_replay_buffer_1d(env, buf_states, buf_actions, buf_max_size, vmax=None):

    # get state and actions in buffer
    n_points = buf_states.shape[0]

    if vmax is None:
        vmax = n_points / 100

    # edges
    x_edges = env.state_space_h[::1]
    y_edges = env.action_space_h[::1]

    H, _, _ = np.histogram2d(buf_states, buf_actions, bins=(x_edges, y_edges))

    # initialize figure
    fig, ax = plt.subplots()
    ax.set_title('Histogram Replay Buffer (State-action)')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    image = ax.imshow(
        H.T,
        vmin=0,
        vmax=vmax,
        interpolation='nearest',
        origin='lower',
        extent=get_state_action_1d_extent(env),
        aspect='auto',
    )
    plt.show()

def initialize_replay_buffer_1d_figure(env, replay_buffer):

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))

    # turn interactive mode on
    plt.ion()

    # get state and actions in buffer
    n_points = replay_buffer.size
    states = replay_buffer.state_buf[:replay_buffer.size, 0]
    actions = replay_buffer.act_buf[:replay_buffer.size, 0]

    # edges
    x_edges = env.state_space_h[::5]
    y_edges = env.action_space_h[::5]

    H, _, _ = np.histogram2d(states, actions, bins=(x_edges, y_edges))

    # frequency table
    ax.set_title('Histogram Replay Buffer (State-action)', fontsize=10)
    ax.set_xlabel('States', fontsize=8)
    ax.set_ylabel('Actions', fontsize=8)

    image = ax.imshow(
        H.T,
        vmin=0,
        vmax=replay_buffer.max_size / 100,
        interpolation='nearest',
        origin='lower',
        extent=get_state_action_1d_extent(env),
        aspect='auto',
    )
    plt.show()

    return (x_edges, y_edges, image)

def update_replay_buffer_1d_figure(env, replay_buffer, tuple_fig):

    # unpack tuple
    x_edges, y_edges, image = tuple_fig

    # get state and actions in buffer
    n_points = replay_buffer.size
    states = replay_buffer.state_buf[:replay_buffer.size, 0]
    actions = replay_buffer.act_buf[:replay_buffer.size, 0]

    H, _, _ = np.histogram2d(states, actions, bins=(x_edges, y_edges))

    # update image
    image.set_data(H.T)

    # pause interval 
    plt.pause(0.1)
