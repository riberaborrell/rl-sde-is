import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_returns_episodes(returns, avg_returns):
    fig, ax = plt.subplots()
    ax.set_title('Returns')
    ax.set_xlabel('Episodes')
    ax.set_ylim(-10, 0)

    plt.plot(returns, label='returns')
    plt.plot(avg_returns, label='running averages returns')
    plt.legend()
    plt.show()

def plot_var_returns_episodes(run_var_returns):
    fig, ax = plt.subplots()
    ax.set_title('Running variance Returns')
    ax.set_xlabel('Episodes')
    #ax.set_ylim(-10, 0)

    plt.semilogy(run_var_returns)
    plt.legend()
    plt.show()


def plot_time_steps_episodes(time_steps, avg_time_steps):
    fig, ax = plt.subplots()
    ax.set_title('Time steps')
    ax.set_xlabel('Episodes')

    plt.plot(time_steps, label='time steps')
    plt.plot(avg_time_steps, label='running averages time steps')
    plt.legend()
    plt.show()

def get_extent(env):
    ''' set extent bounds
    '''

    extent = env.state_space_h[0], env.state_space_h[-1], \
             env.action_space_h[0], env.action_space_h[-1]
    return extent

def plot_frequency(env, n_table):

    fig, ax = plt.subplots()
    ax.set_title('Frequency')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = ax.imshow(
        n_table.T,
        origin='lower',
        extent=get_extent(env),
        cmap=cm.coolwarm,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_q_value_function(env, q_table):

    fig, ax = plt.subplots()
    ax.set_title('Q-value function')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = fig.axes[0].imshow(
        q_table.T,
        origin='lower',
        extent=get_extent(env),
        cmap=cm.viridis,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_advantage_function(env, a_table):

    fig, ax = plt.subplots()
    ax.set_title('Advantage function')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = fig.axes[0].imshow(
        a_table.T,
        origin='lower',
        extent=get_extent(env),
        cmap=cm.plasma,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_value_function(env, value_function, value_function_hjb):

    fig, ax = plt.subplots()
    ax.set_title('Value function')
    ax.set_xlabel('States')

    plt.plot(env.state_space_h, value_function)
    plt.plot(env.state_space_h, -value_function_hjb, label=r'hjb solution')
    plt.plot(env.state_space_h, -value_function_hjb, label=r'hjb solution')
    plt.legend()
    plt.show()

def plot_value_function_actor_critic(env, value_function_actor_critic, value_function_critic,
                                     value_function_hjb):
    fig, ax = plt.subplots()
    ax.set_title('Value function')
    ax.set_xlabel('States')

    plt.plot(env.state_space_h, value_function_actor_critic, label=r'actor-critic: $ V(s) = Q(s, \mu(s; \theta); w)$')
    plt.plot(env.state_space_h, value_function_critic, label=r'critic: $V(s) = max_a Q(s, a; w)$')
    plt.plot(env.state_space_h, -value_function_hjb, label=r'hjb solution')
    plt.legend()
    plt.show()

def plot_stoch_policy(env, action_prob_dists, policy, control_hjb):

    # plot action probability distributions
    fig, ax = plt.subplots()
    ax.set_title('Action Probability distributions')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    im = ax.imshow(
        action_prob_dists.T,
        origin='lower',
        extent=get_extent(env),
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
    plt.legend()
    plt.show()

def plot_det_policy(env, policy, control_hjb):

    fig, ax = plt.subplots()
    ax.set_title('Deterministic Policy')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    plt.plot(env.state_space_h, policy)
    plt.plot(env.state_space_h, control_hjb[:, 0], label=r'hjb solution')
    plt.legend()
    plt.show()

def plot_det_policy_actor_critic(env, policy_actor, policy_critic, control_hjb):
    fig, ax = plt.subplots()
    ax.set_title('Deterministic Policy')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    plt.plot(env.state_space_h, policy_actor, label=r'actor: $\mu(s; \theta)$')
    plt.plot(env.state_space_h, policy_critic, label=r'critic: $\mu(s) = argmax_a Q(s, a; w)$')
    plt.plot(env.state_space_h, control_hjb[:, 0], label=r'hjb solution')
    plt.legend()
    plt.show()

def initialize_det_policy_figure(env, policy, control_hjb):

    fig, ax = plt.subplots()
    ax.set_title('Deterministic Policy')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')
    #ax.set_xlim(env.state_space_low, env.state_space_high)
    #ax.set_ylim(env.action_space_low, env.action_space_high)

    det_policy_line = ax.plot(env.state_space_h, policy)[0]
    ax.plot(env.state_space_h, control_hjb[:, 0], label=r'hjb solution')

    plt.ion()
    plt.legend()
    plt.show()

    return det_policy_line

def update_det_policy_figure(env, policy, line):
    line.set_data(env.state_space_h, policy)
    plt.pause(0.1)


def initialize_episodes_figures(env, n_episodes):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax1, ax2 = axes

    # 
    plt.ion()

    # returns
    ax1.set_title('Return')
    ax1.set_xlabel('Episodes')
    ax1.set_xlim(0, n_episodes)
    ax1.set_ylim(-10, 1)

    ax2.set_title('Time steps')
    ax2.set_xlabel('Episodes')
    ax2.set_xlim(0, n_episodes)
    ax2.set_ylim(0, 1000)

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

def update_episodes_figures(env, returns, avg_returns, time_steps, avg_time_steps, tuples):

    # unpack lines and images
    line_returns, line_avg_returns, line_time_steps, line_avg_time_steps = tuples

    # episodes
    n_episodes = returns.shape[0]
    episodes = np.arange(n_episodes)

    # update plots
    line_returns.set_data(episodes, returns)
    line_avg_returns.set_data(episodes, avg_returns)
    line_time_steps.set_data(episodes, time_steps)
    line_avg_time_steps.set_data(episodes, avg_time_steps)

    # update figure frequency
    plt.pause(0.01)

def initialize_q_learning_figures(env, q_table, v_table, a_table, policy, value_function_hjb, control_hjb):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]

    # 
    plt.ion()

    # q table
    ax1.set_title('Q-value function')
    ax1.set_xlabel('States')
    ax1.set_ylabel('Actions')
    im_q_table = ax1.imshow(
        q_table.T,
        origin='lower',
        extent=get_extent(env),
        cmap=cm.viridis,
        #aspect='auto',
    )

    # value function
    ax2.set_title('Value function')
    ax2.set_xlabel('States')
    line_value_function = ax2.plot(env.state_space_h, v_table)[0]
    ax2.plot(env.state_space_h, -value_function_hjb)

    # a table
    ax3.set_title('Advantage function')
    ax3.set_xlabel('States')
    ax3.set_ylabel('Actions')
    im_a_table = ax3.imshow(
        a_table.T,
        origin='lower',
        extent=get_extent(env),
        cmap=cm.plasma,
        #aspect='auto',
    )

    # control
    ax4.set_title('Value function')
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
    plt.pause(0.01)

def initialize_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic, a_table,
                                    policy_actor, policy_critic, value_function_hjb, control_hjb):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]

    # 
    plt.ion()

    # q table
    ax1.set_title('Q-value function')
    ax1.set_xlabel('States')
    ax1.set_ylabel('Actions')
    im_q_table = ax1.imshow(
        q_table.T,
        origin='lower',
        extent=get_extent(env),
        cmap=cm.viridis,
        #aspect='auto',
    )

    # value function
    ax2.set_title('Value function')
    ax2.set_xlabel('States')
    line_value_f_actor_critic = ax2.plot(env.state_space_h, v_table_actor_critic)[0]
    line_value_f_critic = ax2.plot(env.state_space_h, v_table_critic)[0]
    ax2.plot(env.state_space_h, -value_function_hjb)

    # a table
    ax3.set_title('Advantage function')
    ax3.set_xlabel('States')
    ax3.set_ylabel('Actions')
    im_a_table = ax3.imshow(
        a_table.T,
        origin='lower',
        extent=get_extent(env),
        cmap=cm.plasma,
        #aspect='auto',
    )

    # control
    ax4.set_title('Value function')
    ax4.set_xlabel('States')
    ax4.set_ylabel('Actions')
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
    plt.pause(0.01)
