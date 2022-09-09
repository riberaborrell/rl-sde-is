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

    im = fig.axes[0].imshow(
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
    plt.plot(env.state_space_h, -value_function_hjb)
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


def plot_det_policy(env, policy, control_hjb):

    fig, ax = plt.subplots()
    ax.set_title('Deterministic Policy')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    plt.plot(env.state_space_h, policy)
    plt.plot(env.state_space_h, control_hjb[:, 0], label=r'hjb solution')
    plt.show()

def plot_det_policy_actor_critic(env, policy_actor, policy_critic, control_hjb):
    fig, ax = plt.subplots()
    ax.set_title('Deterministic Policy')
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')

    plt.plot(env.state_space_h, policy_actor, label=r'actor: $\mu(s; \theta)$')
    plt.plot(env.state_space_h, policy_critic, label=r'critic: $\mu(s) = argmax_a Q(s, a; w)$')
    plt.plot(env.state_space_h, control_hjb[:, 0], label=r'hjb solution')
    plt.show()
