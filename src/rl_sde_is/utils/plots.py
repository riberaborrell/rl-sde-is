import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors, cm

import rl_sde_is.utils.figures
from rl_sde_is.utils.numeric import compute_running_mean, compute_running_variance

COLORS_TAB10 = [plt.cm.tab10(i) for i in range(20)]
COLORS_TAB20 = [plt.cm.tab20(i) for i in range(20)]
COLORS_TAB20b = [plt.cm.tab20b(i) for i in range(20)]
COLORS_TAB20c = [plt.cm.tab20c(i) for i in range(20)]
COLORS_FIG = {
    'hjb': 'black',
}

TITLES_FIG = {
    'potential': r'Potential $V(s)$',
    'perturbed-potential': r'Perturbed Potential $\widetilde{V}(s)$',
    'policy': r'Policy $\mu_\theta(s)$',
    'stoch-policy': r'Stochastic policy $\pi_\theta(s)$',
    'value-function': r'Value function $V_\omega(s)$',
    'q-value-function': r'Q-value function $Q_\omega(s, a)$',
    'a-value-function': r'Advantage function $A_\omega(s, a)$',
    'value-rms-error': r'RMS Error of $V(s)$',
    'policy-rms-error': r'RMS Error of $\pi(s)$',
    'policy-l2-error': r'Estimation of $L^2(\mu_\theta)$',
    'loss': r'$\widehat{J}(\mu_\theta)$',
    'returns': r'Return $G_0^\tau$',
    'time-steps': r'TS',
    'ct': r'CT(s)',
}
TITLES_FIG['policy']

def plot_test():
    fig, ax = plt.subplots()
    ax.set_title('Test')
    plt.show()

def get_plot_function(ax, plot_scale):
    if plot_scale == 'linear':
        return ax.plot
    elif plot_scale == 'semilogx':
        return ax.semilogx
    elif plot_scale == 'semilogy':
        return ax.semilogy
    elif plot_scale == 'loglog':
        return ax.loglog
    else:
        raise ValueError('plot_scale must be one of: lineal, semilogx, semilogy, loglog')

def plot_y_per_x(x, y, run_window: int = 1, hlines=None, title: str = '', xlabel: str = '',
                 xlim=None, ylim=None, plot_scale='linear', legend: bool = False, loc=None):

    run_mean_y = compute_running_mean(y, run_window) if run_window > 1 else None
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plot_fn = get_plot_function(ax, plot_scale)
    plot_fn(x, y, c='tab:blue', alpha=0.5)
    if run_window > 1:
        plot_fn(x, run_mean_y, label='running mean', c='tab:blue')
    if hlines:
        for (hline, color, ls, label) in hlines:
            ax.axhline(y=hline, c=color, ls=ls, label=label)
    if legend: plt.legend(loc=loc)
    plt.show()


def plot_y_per_episode(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Episodes', **kwargs)

def plot_y_per_grad_iteration(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Grad. iterations.', **kwargs)

def plot_y_per_time_steps(x, y, **kwargs):
    plot_y_per_x(x, y, xlabel='Time steps', **kwargs)

def plot_y_avg_per_x(x, ys, hlines=None, title: str = '', xlabel: str = '', xlim=None, ylim=None,
                     plot_scale='linear', legend: bool = False, loc: str = 'upper right'):
    y = np.mean(ys, axis=0)
    error = np.sqrt(np.var(ys, axis=0))
    fig, ax = plt.subplots()
    plot_fn = get_plot_function(ax, plot_scale)
    ax.set_title(title, size=20)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plot_fn(x, y, label='Mean')
    ax.fill_between(x, y-error, y+error, alpha=0.4, label='Standard deviation')
    if hlines:
        for (hline, color, ls, label) in hlines:
            ax.axhline(y=hline, c=color, ls=ls, label=label)
    if legend: plt.legend(loc=loc)
    plt.show()

def plot_y_avg_per_episode(x, ys, **kwargs):
    plot_y_avg_per_x(x, ys, xlabel='Episodes', **kwargs)

def plot_y_avg_per_grad_iteration(x, ys, **kwargs):
    plot_y_avg_per_x(x, ys, xlabel='Grad. iterations', **kwargs)

def plot_y_avg_per_time_steps(x, ys, **kwargs):
    plot_y_avg_per_x(x, ys, xlabel='Time steps', **kwargs)

def plot_mean_and_std_per_x(x, mean_y, std_y, hlines=None, title: str = '', xlabel: str = '',
                            xlim=None, ylim=None, plot_scale='linear', legend: bool = False,
                            loc: str = 'upper right'):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plot_fn = get_plot_function(ax, plot_scale)
    plot_fn(x, mean_y, label='mean')
    ax.fill_between(x, mean_y-std_y, mean_y+std_y, alpha=0.4, label='standard deviation')
    if hlines:
        for (hline, std_hline, color, ls, label) in hlines:
            ax.axhline(y=hline, c=color, ls=ls, label=label)
    if legend: plt.legend(loc=loc)
    plt.show()

def plot_mean_and_std_per_episode(x, mean_y, std_y, **kwargs):
    plot_mean_and_std_per_x(x, y, xlabel='Episodes', **kwargs)

def plot_mean_and_std_per_grad_iteration(x, mean_y, std_y, **kwargs):
    plot_mean_and_std_per_x(x, mean_y, std_y, xlabel='Grad. iterations.', **kwargs)

def plot_mean_and_std_per_time_steps(x, mean_y, std_y, **kwargs):
    plot_mean_and_std_per_x(x, mean_y, std_y, xlabel='Time steps', **kwargs)


def plot_ys_per_x(x, ys, run_window: int = 100, title: str = '', xlabel: str = '',
                  xlim=None, ylim=None, labels=None, legend: bool = False, loc=None):
    n_lines = len(ys)
    if labels is None:
        labels = [None for i in range(n_lines)]
    run_mean_ys = np.array([compute_running_mean(y, run_window) for y in ys])
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    for i in range(n_lines):
        plt.plot(run_mean_ys[i], label=labels[i])
    if legend: plt.legend(loc=loc)
    plt.show()

def plot_ys_per_episode(x, ys, **kwargs):
    plot_ys_per_x(x, ys, xlabel='Episodes', **kwargs)

def plot_ys_per_grad_iteration(x, ys, **kwargs):
    plot_ys_per_x(x, ys, xlabel='Grad. iterations', **kwargs)

def plot_ys_per_time_steps(x, ys, **kwargs):
    plot_ys_per_x(x, ys, xlabel='Time steps', **kwargs)

def plot_time_steps_histogram(time_steps):
    n_steps_max = np.max(time_steps)
    x = np.arange(1, n_steps_max+1, 10)
    fig, ax = plt.subplots()
    ax.set_xlabel('m')
    counts, bins = np.histogram(time_steps, bins=x, density=True)
    ax.hist(bins[:-1], bins, weights=counts, alpha=0.5, label=r'Histogram')
    #ax.legend(fontsize=12)
    plt.show()

def plot_value_rms_error_episodes(rms_errors, test_freq_episodes, ylim=None):
    n_test_episodes = rms_errors.shape[0]
    episodes = np.arange(n_test_episodes) * test_freq_episodes
    plot_y_per_episode(episodes, rms_errorsy, title=TITLES_FIG['value-rms-error'], lim=ylim)

def plot_value_rms_error_iterations(rms_errors, test_freq_iterations, ylim=None):
    n_test_iterations = rms_errors.shape[0]
    iterations = np.arange(n_test_iterations) * test_freq_iterations
    plot_y_per_x(iterations, rms_errors, title=TITLES_FIG['value-rms-error'], xlabel='Iterations', ylim=ylim)

def plot_policy_rms_error(x, y, xlabel=None, ylim=None):
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy-rms-error'])
    ax.set_xlabel(xlabel)
    ax.plot(x, y)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.show()

def plot_policy_rms_error_episodes(rms_errors, test_freq_episodes, ylim=None):
    n_test_episodes = rms_errors.shape[0]
    episodes = np.arange(n_test_episodes) * test_freq_episodes
    plot_policy_rms_error(episodes, rms_errors, xlabel='Episodes', ylim=ylim)

def plot_policy_rms_error_iterations(rms_errors, test_freq_iterations, ylim=None):
    n_test_iterations = rms_errors.shape[0]
    iterations = np.arange(n_test_iterations) * test_freq_iterations
    plot_policy_rms_error(iterations, rms_errors, xlabel='Iterations', ylim=ylim)

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
    #ax.set_title(TITLES_FIG['policy-l2-error'])
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
    #ax.set_title(TITLES_FIG['policy-l2-error'])
    ax.set_xlabel('Trajectories')
    if episodes is not None:
        plt.semilogy(episodes, l2_errors)
    else:
        plt.semilogy(l2_errors)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.show()

def get_state_action_1d_extent(env):
    ''' set extent bounds for 1d state space in the x-axis and
        1d action space in the y-axis
    '''

    state_space_h = env.state_space_h.squeeze()
    action_space_h = env.action_space_h.squeeze()
    extent = state_space_h[0] - env.h_state/2, state_space_h[-1] + env.h_state/2, \
             action_space_h[0] - env.h_action/2, action_space_h[-1] + env.h_action/2
    return extent

def get_state_2d_extent(env):
    ''' set extent bounds for 2d state space
    '''

    extent = env.state_space_h[0, 0, 0] - env.h_state/2, env.state_space_h[-1, -1, 0] + env.h_state/2, \
             env.state_space_h[0, 0, 1] - env.h_state/2, env.state_space_h[-1, -1, 1] + env.h_state/2
    return extent

def plot_episode_states_1d(env, ep_states, loc=None):

    # compute potential at each state
    ep_pot = env.potential(ep_states)

    # draw target set
    target_set = patches.Rectangle(
            (1, 0), 2, 15, lw=1.5, edgecolor='black', facecolor='tab:orange', alpha=0.5
    )

    # compute potential on the grid
    potential = env.potential(env.state_space_h)

    fig, ax = plt.subplots()
    ax.set_title(r'Trajectory')
    ax.set_xlabel(r'States')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 4)
    ax.text(1.12, 2.2, r'Target set', size=13, rotation=0.)
    ax.add_patch(target_set)
    ax.plot(env.state_space_h, potential, label=r'Potential $V_\alpha$')
    ax.scatter(ep_states[::1], ep_pot[::1], alpha=.1, c='black', marker='o', s=100)
    plt.legend(loc=loc)
    plt.show()

def plot_episode_states_2d(env, ep_states):

    # draw target set
    target_set = patches.Rectangle(
            (1, 1), 1, 1, lw=1.5, edgecolor='black', facecolor='tab:orange', alpha=0.5
    )

    # compute potential on the grid
    flat_state_space_h = env.state_space_h.reshape(env.n_states, env.d)
    potential = env.potential(flat_state_space_h).reshape(env.state_space_h.shape[:-1])

    fig, ax = plt.subplots()
    ax.set_title(r'Trajectory')
    ax.set_xlabel(r'$s_1$')
    ax.set_xlabel(r'$s_2$')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.add_patch(target_set)
    ax.imshow(
        potential,
        origin='lower',
        extent=get_state_2d_extent(env),
        cmap=cm.coolwarm,
    )
    ax.scatter(ep_states[:, 0], ep_states[:, 1], alpha=.1, c='black', marker='o', s=100)
    plt.show()

def plot_reward_table(env, r_table):

    fig, ax = plt.subplots()
    ax.set_title(r'$r(s, a)$')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = ax.imshow(
        r_table.T,
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.coolwarm,
        aspect='auto',
    )

    # add space for colour bar
    #fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im)#, cax=cbar_ax)

    plt.show()

def plot_reward_following_policy(env, rewards):

    fig, ax = plt.subplots()
    ax.set_title('$r(s, \mu(s))$')
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])
    ax.plot(env.state_space_h, rewards)
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


def get_colors_and_labels(iterations=None, n_lines=None, iter_str='traj.', hjb=True):

    # fill epochs if not provided
    if iterations is None and n_lines is None:
        n_lines = 6
    elif iterations is None and n_lines is not None:
        iterations = [i for i in range(n_lines)] if iterations is None else iterations
    elif iterations is not None and n_lines is None:
        n_lines = len(iterations)
        assert 2 <= n_lines <= 6, 'The maximum number of lines is 6'
    elif iterations is not None and n_lines is not None:
        assert len(iterations) == n_lines, 'The number of iterations must match the number of lines'

    # set colors and labels
    colors = [
            'tab:blue',
            COLORS_TAB20b[15],
            COLORS_TAB20b[14],
            COLORS_TAB20b[13],
            COLORS_TAB20b[12],
            COLORS_TAB20b[16],
    ]
    if n_lines <= 5:
        colors.pop(5)
    if n_lines <= 4:
        colors.pop(3)
    if n_lines <= 3:
        colors.pop(2)
    if n_lines <= 2:
        colors.pop(1)

    labels = ['initial'] \
           + ['{} {:d}'.format(iter_str, i) for i in iterations[1:]]
    if hjb:
        colors.append(COLORS_FIG['hjb'])
        labels.append('hjb')

    return colors, labels

def plot_q_value_function_1d(env, q_table, vmin=None, file_path=None):

    fig, ax = plt.subplots()
    #ax.set_title(TITLES_FIG['q-value-function'])
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    #ax.plot(np.nan, alpha=0., label=r'bla')
    #ax.legend()

    #im = fig.axes[0].imshow(
    im = ax.imshow(
        q_table.T,
        vmin=vmin,
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

    plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12, top=0.98)

    if file_path is not None:
        plt.savefig(file_path, format='pdf')
    else:
        plt.show()

def plot_advantage_function_1d(env, a_table, policy_opt=None, policy_critic=None,
                               vmin=None, file_path=None):

    fig, ax = plt.subplots()
    #ax.set_title(TITLES_FIG['a-value-function'])
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    if policy_critic is not None:
        ax.plot(env.state_space_h, policy_critic, c='grey', ls='--', label=r'${argmax}_a A_\omega^h(s, a)$')

    if policy_opt is not None:
        ax.plot(env.state_space_h, policy_opt, ls=':', c='black', label=r'hjb')

    im = ax.imshow(
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

    ax.legend()

    if file_path is not None:
        plt.savefig(file_path, format='pdf')
    else:
        plt.show()

def plot_value_function_1d(env, value_function, value_function_opt=None, ylim=None,
                           legend: bool = False, loc=None):

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['value-function'])
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])
    if ylim: ax.set_ylim(ylim)
    ax.plot(env.state_space_h, value_function)
    if value_function_opt is not None:
        ax.plot(env.state_space_h, value_function_opt, label=r'hjb', ls=':', c='black')
    ax.legend(loc=loc)
    plt.show()

def plot_value_function_2d(env, value_function, value_function_opt, levels=10,
                           isolines=True, title: str = '', file_path=None):

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(r'$s_1$')
    ax.set_ylabel(r'$s_2$')

    cs = ax.contourf(
        env.state_space_h[:, :, 0],
        env.state_space_h[:, :, 1],
        value_function,
        levels=levels,
        vmin=np.min(value_function_opt),
        vmax=np.max(value_function_opt),
        origin='lower',
        extent=get_state_2d_extent(env),
        cmap='Blues_r',
    )
    if isolines: ax.contour(cs, colors='k')

    # add space for colour bar
    #fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    #fig.colorbar(cs, cax=cbar_ax)
    fig.colorbar(cs)

    plt.savefig(file_path, format='pdf') if file_path is not None else plt.show()

def plot_value_function_1d_actor_critic(env, value_function_critic_initial, value_function_critic,
                                        value_function_actor_critic, value_function_opt,
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
    ax.plot(x, value_function_opt, label=r'hjb', c=COLORS_FIG['hjb'], ls=':')
    ax.legend(loc=loc)
    plt.show()

def plot_categorical_stoch_policy_1d(env, action_prob_dists, policy, policy_opt, loc=None):

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
    ax.set_title('Actions sampled following policy')
    ax.set_xlabel('States')
    ax.scatter(env.state_space_h, policy)
    ax.plot(env.state_space_h, policy_opt, c=COLORS_FIG['hjb'], ls=':', label=r'optimal')
    plt.legend(loc=loc)
    plt.show()

def plot_gaussian_stoch_policy_1d(env, mu, sigma_sq, policy_opt):

    # plot mu
    fig, ax = plt.subplots()
    ax.set_title(r'Mean $m_\theta(s)$')
    ax.set_xlabel('States')
    ax.plot(env.state_space_h, mu)
    ax.plot(env.state_space_h, policy_opt, c=COLORS_FIG['hjb'], ls=':', label=r'optimal')
    plt.show()

    # plot sigma
    fig, ax = plt.subplots()
    ax.plot(env.state_space_h, sigma_sq)
    ax.set_title(r'Sigma $\Sigma_\theta(s)$')
    ax.set_xlabel('States')
    plt.show()

    # plot mu with error
    x = env.state_space_h.squeeze()
    y = mu
    error = np.sqrt(sigma_sq)
    fig, ax = plt.subplots()
    ax.set_title(r'Mean $m_\theta(s)$')
    ax.set_xlabel('State space')
    ax.plot(x, y, label='mu')
    ax.fill_between(x, y-error, y+error, alpha=0.5, label='standard deviation')
    ax.plot(env.state_space_h, policy_opt, c=COLORS_FIG['hjb'], ls=':', label=r'optimal')
    plt.show()

def plot_det_policy_1d(env, policy, policy_opt=None, loc=None):

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])

    ax.plot(env.state_space_h, policy)
    if policy_opt is not None:
        ax.plot(env.state_space_h, policy_opt, c=COLORS_FIG['hjb'], ls=':', label=r'optimal')
    ax.legend(loc=loc)
    plt.show()

def plot_det_policies_1d(env, policies, policy_opt=None, labels=None, colors=None,
                         ylim=None, loc='upper right', file_path=None):

    n_policies = len(policies)

    if labels is None:
        labels = ['' for i in range(n_policies + 1)]

    if colors is None:
        colors = [None for i in range(n_policies + 1)]

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    #ax.set_xlabel('x')
    ax.set_xlabel('States')
    #ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])
    ax.set_xlim(-1.8, 1.8)
    if ylim is not None:
        ax.set_ylim(ylim)

    x = env.state_space_h
    for i in range(n_policies):
        ax.plot(x, policies[i], c=colors[i], label=labels[i])
    if policy_opt is not None:
        ax.plot(x, policy_opt, c=COLORS_FIG['hjb'], ls=':', label=labels[i+1])

    if labels[0]:
        #plt.legend(loc=loc, fontsize=10)
        ax.legend(loc=loc, fontsize=12)

    if file_path is not None:
        plt.savefig(file_path, format='pdf')
    else:
        plt.show()

def plot_det_policies_1d_black_and_white(env, policies, policy_opt):
    n_policies = policies.shape[0]
    cmap = cm.get_cmap('Greys')
    colors = cmap(np.linspace(0, 1, n_policies))

    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])

    for i in range(n_policies):
        ax.plot(env.state_space_h, policies[i], c=colors[i])
    ax.plot(env.state_space_h, policy_opt, c='black', ls='-.')
    #ax.set_ylim(-3, 3)
    plt.show()

def plot_ys_1d(env, ys, y_opt=None, title: str = '', xlim=None, ylim=None, labels=None,
               colors=None, legend: bool = False, loc='upper right', file_path=None):

    n_lines = len(ys)

    if labels is None:
        labels = ['' for i in range(n_lines + 1)]

    if colors is None:
        colors = [None for i in range(n_lines + 1)]

    x = env.state_space_h

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('States')
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    for i in range(n_lines):
        ax.plot(x, ys[i], c=colors[i], label=labels[i])

    if y_opt is not None:
        ax.plot(x, y_opt, c=colors[i+1], ls=':', label=labels[i+1])

    if legend: plt.legend(loc=loc, fontsize=12)

    if file_path is not None:
        plt.savefig(file_path, format='pdf')
    else:
        plt.show()


def initialize_det_policy_1d_figure(env, policy, policy_critic=None, policy_opt=None, title: str = ''):

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))

    # turn interactive mode on
    plt.ion()

    ax.set_title(title) if title else ax.set_title(TITLES_FIG['policy'])
    ax.set_xlabel('States', fontsize=8)
    ax.set_xlim(env.state_space_bounds)
    ax.set_ylim(env.action_space_bounds)
    det_policy_line = ax.plot(env.state_space_h, policy)[0]
    if policy_opt is not None:
        ax.plot(env.state_space_h, policy_opt, c=COLORS_FIG['hjb'], ls=':', label=r'hjb')
    if policy_critic is not None:
        ax.plot(env.state_space_h, policy_critic, c='grey', ls='--', label=r'$argmax_a Q(s, a)$')

    plt.legend()
    plt.show()

    return det_policy_line

def update_det_policy_1d_figure(env, policy, line):
    line.set_data(env.state_space_h, policy)
    plt.pause(0.1)

def initialize_gaussian_policy_1d_figure(env, mean, sigma, policy_opt=None, title: str = ''):

    # initialize figure
    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes

    # turn interactive mode on
    plt.ion()

    ax1.set_title(r'Mean $\mu_\theta$')
    ax1.set_xlabel('States', fontsize=8)
    ax1.set_xlim(env.state_space_bounds)
    ax1.set_ylim(env.action_space_bounds)
    mean_line = ax1.plot(env.state_space_h, mean)[0]
    if policy_opt is not None:
        ax1.plot(env.state_space_h, policy_opt, c=COLORS_FIG['hjb'], ls=':', label=r'hjb')

    ax2.set_title(r'Sigma $\Sigma_\theta$')
    ax2.set_xlabel('States', fontsize=8)
    ax2.set_xlim(env.state_space_bounds)
    ax2.set_ylim(0, 10)
    sigma_line = ax2.plot(env.state_space_h, sigma)[0]

    plt.legend()
    plt.show()

    return mean_line, sigma_line

def update_gaussian_policy_1d_figure(env, mean, sigma, lines):
    mean_line, sigma_line = lines
    mean_line.set_data(env.state_space_h, mean)
    sigma_line.set_data(env.state_space_h, sigma)
    plt.pause(0.1)

def canvas_det_policy_1d_figure(env, policies, policy_opt):

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.show()
    fig.canvas.draw()

    for i, policy in enumerate(policies):

        # update title
        fig.suptitle('iteration: {:d}'.format(i), fontsize='x-large')

        ax.cla()
        ax.set_title(TITLES_FIG['policy'])
        ax.set_xlabel('States', fontsize=8)
        ax.set_xlim(env.state_space_bounds)
        ax.set_ylim(env.action_space_bounds)
        ax.plot(env.state_space_h, policy)[0]
        ax.plot(env.state_space_h, policy_opt, label=r'hjb', c=COLORS_FIG['hjb'])

        #plt.legend()
        #plt.show()

        # pause
        plt.pause(0.01)

        # draw
        fig.canvas.draw()


def plot_det_policy_1d_actor_critic(env, policy_actor, policy_critic,
                                    policy_actor_initial=None, policy_critic_initial=None,
                                    policy_opt=None, ylim=None, loc=None):
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlabel('States')
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])
    if ylim is not None:
        ax.set_ylim(ylim)

    x = env.state_space_h
    if policy_actor_initial is not None:
        ax.plot(x, policy_actor_initial, label=r'initial actor', c='tab:blue')
    if policy_critic_initial is not None:
        ax.plot(x, policy_critic_initial, label=r'initial actor', c='tab:red')
    ax.plot(x, policy_actor, label=r'actor: $\mu_\theta(s)$', c='tab:green')
    ax.plot(x, policy_critic, label=r'critic: $\mu_\omega(s) = argmax_a Q_\omega(s, a)$',
            c='tab:orange')
    if policy_opt is not None:
        ax.plot(x, policy_opt, c='black', ls=':', label=r'hjb')
    ax.legend(loc=loc)
    plt.show()

def plot_det_policy_2d(env, policy, policy_opt, title: str = '',
                       vmin=None, vmax=None, file_path=None):
    X = env.state_space_h[:, :, 0]
    Y = env.state_space_h[:, :, 1]
    U = policy[:, :, 0]
    V = policy[:, :, 1]
    X, Y, U, V = coarse_quiver_arrows(U, V, X, Y, l=25)

    U_hjb = policy_opt[:, :, 0]
    V_hjb = policy_opt[:, :, 1]
    _, _, U_hjb, V_hjb = coarse_quiver_arrows(U_hjb, V_hjb, l=25)

    # initialize figure
    fig, ax = plt.subplots()
    ax.set_title(TITLES_FIG['policy']) if not title else ax.set_title(title)
    ax.set_xlabel(r'$s_1$')
    ax.set_ylabel(r'$s_2$')
    ax.set_xlim(env.state_space_bounds[0, 0], env.state_space_bounds[0, 1])
    ax.set_ylim(env.state_space_bounds[1, 0], env.state_space_bounds[1, 1])

    # initialize norm object
    C = np.sqrt(U**2 + V**2)
    C_hjb = np.sqrt(U_hjb**2 + V_hjb**2)
    vmin = np.min(C_hjb) if vmin is None else vmin
    vmax = np.max(C_hjb) if vmax is None else vmax
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

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

    if file_path is not None:
        plt.savefig(file_path, format='pdf')
    else:
        plt.show()

def initialize_stoch_policy_1d_figure(env, policy, policy_opt):

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))

    # turn interactive mode on
    plt.ion()

    ax.set_title(TITLES_FIG['stoch-policy'])
    ax.set_xlabel('States', fontsize=8)
    ax.set_xlim(env.state_space_bounds)
    ax.set_ylim(env.action_space_bounds)
    sc = ax.scatter(env.state_space_h, policy)
    ax.plot(env.state_space_h, policy_opt, label=r'hjb', c=COLORS_FIG['hjb'])

    plt.legend()
    plt.show()

    return sc

def update_stoch_policy_1d_figure(env, policy, sc):
    sc.set_offsets(np.c_[env.state_space_h, policy])
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
    ax.set_title(TITLES_FIG['policy'])
    ax.set_xlabel(r'$s_1$', fontsize=8)
    ax.set_ylabel(r'$s_2$', fontsize=8)
    ax.set_xlim(env.observation_space.low[0], env.observation_space.high[0])
    ax.set_ylim(env.observation_space.low[1], env.observation_space.high[1])

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
        angles='xy',
        pivot='tail',
        scale=1,
        scale_units='xy',
        units='xy',
        width=0.01,
        norm=norm,
        cmap=cm.viridis,
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
    Q_policy.set_clim(vmin=C.min(), vmax=C.max())

    # update figure frequency
    plt.pause(0.1)

def canvas_det_policy_2d_figure(env, data, backup_episodes, policy_opt, scale=1.0, width=0.005):
    import torch
    from rl_sde_is.td3_core import load_backup_models
    from rl_sde_is.approximate_methods import compute_det_policy_actions

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))
    #ax.set_title(TITLES_FIG['policy'])
    plt.suptitle(TITLES_FIG['policy'], fontsize=10)
    ax.set_xlabel(r'$s_1$', fontsize=8)
    ax.set_ylabel(r'$s_2$', fontsize=8)
    ax.set_xlim(env.state_space_bounds[0], env.state_space_bounds[0])
    ax.set_ylim(env.state_space_bounds[0], env.state_space_bounds[0])

    # turn interactive mode on
    plt.ion()

    X = env.state_space_h[:, :, 0]
    Y = env.state_space_h[:, :, 1]

    U_hjb = policy_opt[:, :, 0]
    V_hjb = policy_opt[:, :, 1]
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
        angles='xy',
        scale=scale,
        scale_units='xy',
        width=width,
        cmap=cm.viridis,
        norm=norm,
    )

    # add space for color bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(Q_policy, cax=cbar_ax)

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

        # update title
        ax.set_title('Episode: {:d}'.format(ep))

        # update plots
        Q_policy.set_UVC(U, V, C)
        Q_policy.set_clim(vmin=C.min(), vmax=C.max())

        # update figure frequency
        plt.pause(0.5)


def initialize_return_and_time_steps_figures(env, n_episodes):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 4))
    ax1, ax2 = axes
    fig.tight_layout(pad=2.5)

    # turn interactive mode on
    plt.ion()

    # returns
    ax1.set_title('Return')
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

def initialize_value_function_1d_figure(env, value_function, value_function_opt=None):

    # initialize figure
    fig, ax = plt.subplots()

    # turn interactive mode on
    plt.ion()

    ax.set_title(TITLES_FIG['value-function'], fontsize=10)
    ax.set_xlabel('States', fontsize=8)
    ax.set_xlim(env.state_space_h[0], env.state_space_h[-1])

    det_v_function_line = ax.plot(env.state_space_h, value_function)[0]
    if value_function_opt is not None:
        ax.plot(env.state_space_h, value_function_opt, c=COLORS_FIG['hjb'], ls=':', label=r'hjb')

    plt.legend()
    plt.show()

    return det_v_function_line

def update_value_function_1d_figure(env, value_function, line):
    line.set_data(env.state_space_h, value_function)
    #line.set_clim(vmin=value_function.min(), vmax=value_function.max())
    plt.pause(0.1)

def initialize_qvalue_function_1d_figure(env, q_table):

    # initialize figure
    fig, ax = plt.subplots()

    # turn interactive mode on
    plt.ion()

    # a table
    ax.set_title(TITLES_FIG['q-value-function'])
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')
    im_q_table = ax.imshow(
        q_table.T,
        cmap=cm.viridis,
        vmin=q_table.min(),
        vmax=q_table.max(),
        aspect='auto',
        origin='lower',
        extent=get_state_action_1d_extent(env),
    )
    # colorbar
    plt.colorbar(im_q_table, ax=ax)

    plt.show()
    return im_q_table

def update_imshow_figure(env, X, im):

    im.set_data(X.T)
    im.set_clim(vmin=X.min(), vmax=X.max())

    # update figure frequency
    plt.pause(0.1)

def initialize_advantage_function_1d_figure(env, a_table, policy_opt, policy_critic=None):

    # initialize figure
    fig, ax = plt.subplots()

    # turn interactive mode on
    plt.ion()

    # a table
    ax.set_title(TITLES_FIG['a-value-function'])
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')
    im = ax.imshow(
        a_table.T,
        cmap=cm.plasma,
        vmin=a_table.min(),
        vmax=a_table.max(),
        aspect='auto',
        origin='lower',
        extent=get_state_action_1d_extent(env),
    )
    ax.plot(env.state_space_h, policy_opt, c='black', ls=':')
    if policy_critic is not None:
        line = ax.plot(env.state_space_h, policy_critic, c='grey', ls='--')[0]
    \
    # colorbar
    plt.colorbar(im, ax=ax)

    plt.show()
    if policy_critic is not None:
        return im, line
    else:
        return im

def update_advantage_function_1d_figure(env, a_table, im, policy_critic=None, line=None):

    im.set_data(a_table.T)
    im.set_clim(vmin=a_table.min(), vmax=a_table.max())
    if policy_critic is not None:
        line.set_data(env.state_space_h, policy_critic)

    # update figure frequency
    plt.pause(0.1)

def initialize_tabular_figures(env, q_table, v_table, a_table,
                               policy, value_function_opt, policy_opt):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]

    # turn interactive mode on
    plt.ion()

    # q table
    ax1.set_title(r'Q-value function $Q(s, a)$')
    ax1.set_xlabel('States')
    ax1.set_ylabel('Actions')
    im_q_table = ax1.imshow(
        q_table.T,
        cmap=cm.viridis,
        vmin=q_table.min(),
        vmax=q_table.max(),
        aspect='auto',
        origin='lower',
        extent=get_state_action_1d_extent(env),
    )

    # value function
    ax2.set_title('Value function $V(s)$')
    ax2.set_xlabel('States')
    line_value_function = ax2.plot(env.state_space_h, v_table)[0]
    ax2.plot(env.state_space_h, value_function_opt, c='black', ls=':')

    # a table
    ax3.set_title(r'Advantage function $A(s, a)$')
    ax3.set_xlabel('States')
    ax3.set_ylabel('Actions')
    im_a_table = ax3.imshow(
        a_table.T,
        cmap=cm.plasma,
        aspect='auto',
        vmin=a_table.min(),
        vmax=a_table.max(),
        origin='lower',
        extent=get_state_action_1d_extent(env),
    )

    # control
    ax4.set_title(r'Deterministic policy $\mu(s)$')
    ax4.set_xlabel('States')
    ax4.set_ylabel('Actions')
    ax4.set_xlim(env.state_space_h[0], env.state_space_h[-1])
    ax4.set_ylim(env.action_space_h[0], env.action_space_h[-1])
    line_control = ax4.plot(env.state_space_h, policy)[0]
    ax4.plot(env.state_space_h, policy_opt, c='black', ls=':')

    # colorbars                                                                 
    plt.colorbar(im_q_table, ax=ax1)
    plt.colorbar(im_a_table, ax=ax3)

    plt.show()
    return (im_q_table, line_value_function, im_a_table, line_control)

def update_tabular_figures(env, q_table, v_table, a_table, policy, tuples):

    # unpack lines and images
    im_q_table, line_value_function, im_a_table, line_policy = tuples

    # update plots
    im_q_table.set_data(q_table.T)
    im_q_table.set_clim(vmin=q_table.min(), vmax=q_table.max())
    line_value_function.set_data(env.state_space_h, v_table)
    im_a_table.set_data(a_table.T)
    im_a_table.set_clim(vmin=a_table.min(), vmax=a_table.max())
    line_policy.set_data(env.state_space_h, policy)

    # update figure frequency
    plt.pause(0.1)

def initialize_frequency_figure(env, n_table):

    # initialize figure
    fig, ax = plt.subplots()

    # turn interactive mode on
    plt.ion()

    ax.set_title('Frequency')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')
    im_n_table = ax.imshow(
        n_table.T,
        aspect='auto',
        #interpolation='nearest',
        vmin=n_table.min(),
        vmax=n_table.max(),
        origin='lower',
        extent=get_state_action_1d_extent(env),
        cmap=cm.plasma,
    )
    plt.show()
    return im_n_table

def update_frequency_figure(env, n_table, im_n_table):

    # update image
    im_n_table.set_data(n_table.T)
    im_n_table.set_clim(vmin=n_table.min(), vmax=n_table.max())

    # pause interval
    plt.pause(0.1)

def initialize_actor_critic_1d_figures(env, q_table, v_table_actor_critic, v_table_critic, a_table,
                                       policy_actor, policy_critic, value_function_opt, policy_opt):

    ''' initialize q-value function, value function, a-value function and policy figure '''

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
    ax1.set_xlim(env.state_space_bounds)
    ax1.set_ylim(env.action_space_bounds)
    im_q_table = ax1.imshow(
        q_table.T,
        cmap=cm.viridis,
        aspect='auto',
        vmin=q_table.min(),
        vmax=q_table.max(),
        origin='lower',
        extent=get_state_action_1d_extent(env),
    )

    # value function
    ax2.set_title('Value function', fontsize=10)
    ax2.set_xlabel('States', fontsize=8)
    ax2.set_xlim(env.state_space_bounds)
    line_value_f_actor_critic = None #ax2.plot(env.state_space_h, v_table_actor_critic)[0]
    line_value_f_critic = ax2.plot(env.state_space_h, v_table_critic, c='tab:orange')[0]
    ax2.plot(env.state_space_h, value_function_opt, c=COLORS_FIG['hjb'], ls=':')

    # a table
    ax3.set_title(TITLES_FIG['a-value-function'], fontsize=10)
    ax3.set_xlabel('States', fontsize=8)
    ax3.set_ylabel('Actions', fontsize=8)
    ax3.set_xlim(env.state_space_bounds)
    ax3.set_ylim(env.action_space_bounds)
    im_a_table = ax3.imshow(
        a_table.T,
        cmap=cm.plasma,
        aspect='auto',
        vmin=a_table.min(),
        vmax=a_table.max(),
        origin='lower',
        extent=get_state_action_1d_extent(env),
    )

    # control
    ax4.set_title(TITLES_FIG['policy'], fontsize=10)
    ax4.set_xlabel('States', fontsize=8)
    ax4.set_ylabel('Actions', fontsize=8)
    ax4.set_xlim(env.state_space_bounds)
    ax4.set_ylim(env.action_space_bounds[0], env.action_space_bounds[1])
    line_policy_actor = ax4.plot(env.state_space_h, policy_actor, c='tab:blue')[0]
    line_policy_critic = ax4.plot(env.state_space_h, policy_critic, c='tab:orange')[0]
    ax4.plot(env.state_space_h, policy_opt, c=COLORS_FIG['hjb'], ls=':', label=r'optimal')

    # colorbars                                                                 
    plt.colorbar(im_q_table, ax=ax1)
    plt.colorbar(im_a_table, ax=ax3)

    plt.show()
    return (im_q_table, line_value_f_actor_critic, line_value_f_critic, im_a_table,
            line_policy_actor, line_policy_critic)

def update_actor_critic_1d_figures(env, q_table, v_table_actor_critic, v_table_critic, a_table,
                                   policy_actor, policy_critic, tuples):

    ''' update q-value function, value function, a-value function and policy figure '''
    # unpack lines and images
    im_q_table, line_value_f_actor_critic, line_value_f_critic, im_a_table, \
            line_policy_actor, line_policy_critic = tuples

    # update plots
    im_q_table.set_data(q_table.T)
    im_q_table.set_clim(vmin=q_table.min(), vmax=q_table.max())
    #line_value_f_actor_critic.set_data(env.state_space_h, v_table_actor_critic)
    line_value_f_critic.set_data(env.state_space_h, v_table_critic)
    im_a_table.set_data(a_table.T)
    im_a_table.set_clim(vmin=a_table.min(), vmax=a_table.max())
    line_policy_actor.set_data(env.state_space_h, policy_actor)
    line_policy_critic.set_data(env.state_space_h, policy_critic)

    # update figure frequency
    plt.pause(0.1)

def canvas_actor_critic_1d_figures(env, data, value_function_opt, policy_opt,
                                   n_episodes=int(1e4), step=100):
    from rl_sde_is.td3_core import load_backup_models
    from rl_sde_is.approximate_methods import compute_tables_critic_1d, compute_tables_actor_critic_1d

    # get models
    actor = data['actor']
    critic = data['critic1']

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2)#, figsize=(8, 6))
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]
    fig.tight_layout(pad=4.0)

    # turn interactive mode on
    plt.ion()

    # q table
    ax1.set_title(TITLES_FIG['q-value-function'])
    ax1.set_xlabel('States')
    ax1.set_ylabel('Actions')
    ax1.set_xlim(env.state_space_bounds)
    ax1.set_ylim(env.action_space_bounds)

    # a table
    ax3.set_title(TITLES_FIG['a-value-function'])
    ax3.set_xlabel('States')
    ax3.set_ylabel('Actions')
    ax3.set_xlim(env.state_space_bounds)
    ax3.set_ylim(env.action_space_bounds)

    # looop to update figures
    episodes = np.arange(0, n_episodes + step, step)
    for ep in episodes:

        # load models
        load_backup_models(data, ep)
        q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
        v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)

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
        ax2.set_xlim(env.state_space_bounds)
        ax2.plot(env.state_space_h, v_table_actor_critic)
        ax2.plot(env.state_space_h, v_table_critic)
        ax2.plot(env.state_space_h, value_function_opt, c='black', ls=':')

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
        ax4.set_xlim(env.state_space_bounds)
        ax4.set_ylim(env.action_space_bounds[0], env.action_space_bounds[1])
        ax4.plot(env.state_space_h, policy_actor)
        ax4.plot(env.state_space_h, policy_critic)
        ax4.plot(env.state_space_h, policy_opt, c='black', ls=':')

        # pause
        plt.pause(1.0)

        # draw
        fig.canvas.draw()

def plot_replay_buffer_1d(env, buf_states, buf_actions, file_path=None):

    # get state and actions in buffer
    n_points = buf_states.shape[0]

    # edges
    x_edges = env.state_space_h[::1].squeeze()
    y_edges = env.action_space_h[::1].squeeze()

    H, _, _ = np.histogram2d(buf_states, buf_actions, bins=(x_edges, y_edges))
    H /= n_points

    # initialize figure
    fig, ax = plt.subplots()
    ax.set_title('Histogram Replay Buffer (State-action)')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    image = ax.imshow(
        H.T,
        aspect='auto',
        interpolation='nearest',
        vmin=H.min(),
        vmax=H.max(),
        origin='lower',
        extent=get_state_action_1d_extent(env),
    )

    if file_path is not None:
        plt.savefig(file_path, format='pdf')
    else:
        plt.show()

def plot_replay_buffer_states_2d(env, buf_states, file_path=None):

    # get state and actions in buffer
    n_points = buf_states.shape[0]

    # edges
    x_edges = env.state_space_h[:, 0, 0]
    y_edges = env.state_space_h[:, 0, 0]

    H, _, _ = np.histogram2d(buf_states[:, 0], buf_states[:, 1], bins=(x_edges, y_edges))
    H /= n_points

    # initialize figure
    fig, ax = plt.subplots()
    ax.set_title('Histogram Replay Buffer (States)')
    ax.set_xlabel(r'$s_1$')
    ax.set_ylabel(r'$s_2$')

    image = ax.imshow(
        H.T,
        aspect='auto',
        interpolation='nearest',
        vmin=H.min(),
        vmax=H.max(),
        origin='lower',
        extent=get_state_2d_extent(env),
    )

    if file_path is not None:
        plt.savefig(file_path, format='pdf')
    else:
        plt.show()

def initialize_replay_buffer_1d_figure(env, replay_buffer):

    # initialize figure
    fig, ax = plt.subplots(figsize=(5, 4))

    # turn interactive mode on
    plt.ion()

    # get state and actions in buffer
    n_points = replay_buffer.size
    states = replay_buffer.states[:replay_buffer.size, 0]

    if replay_buffer.is_action_continuous:
        actions = replay_buffer.actions[:replay_buffer.size, 0]
    else:
        actions = env.action_space_h[replay_buffer.actions[:replay_buffer.size]]

    # edges
    x_edges = env.state_space_h[::5].squeeze()
    y_edges = env.action_space_h[::5].squeeze()

    H, _, _ = np.histogram2d(states, actions, bins=(x_edges, y_edges))
    H /= n_points

    # frequency table
    ax.set_title('Histogram Replay Buffer (State-action)', fontsize=10)
    ax.set_xlabel('States', fontsize=8)
    ax.set_ylabel('Actions', fontsize=8)

    image = ax.imshow(
        H.T,
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
    states = replay_buffer.states[:replay_buffer.size, 0]

    if replay_buffer.is_action_continuous:
        actions = replay_buffer.actions[:replay_buffer.size, 0]
    else:
        actions = env.action_space_h[replay_buffer.actions[:replay_buffer.size]]

    H, _, _ = np.histogram2d(states, actions, bins=(x_edges, y_edges))
    H /= n_points

    # update image
    image.set_data(H.T)
    image.set_clim(vmin=H.min(), vmax=H.max())

    # pause interval 
    plt.pause(0.1)


def get_key_array(datas, key):
    ''' Get array from list of dictionaries with the same key.
        datas: list of dictionaries
        key: entry key
    '''

    n_results = len(datas)

    # create empty array
    array = np.empty((n_results,) + datas[1][key].shape)

    for idx, data in enumerate(datas):
        array[idx] = data[key]

    return array
