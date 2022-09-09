import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_epsilon_greedy_action(env, q_table, idx_state, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        idx_action = np.argmax(q_table[idx_state])

    # pick random action (exploration)
    else:
        idx_action = np.random.choice(np.arange(env.n_actions))

    action = env.action_space_h[[idx_action]]

    return idx_action, action

def get_epsilon_greedy_actions_vectorized(env, q_table, idx_states, epsilon):

    # get batch size
    batch_size = idx_states.shape[0]

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        idx_actions = np.argmax(q_table[idx_states], axis=1)

    # pick random action (exploration)
    else:
        idx_actions = np.random.choice(np.arange(env.n_actions), batch_size)

    actions = env.action_space_h[idx_actions].reshape(batch_size, 1)

    return idx_actions, actions

def set_epsilons(eps_type, **kwargs):

    # constant sequence
    if eps_type == 'constant':
        epsilons = get_epsilons_constant(
            n_episodes=kwargs['n_episodes'],
            eps_init=kwargs['eps_init'],
        )

    # linear decaying sequence
    elif eps_type == 'linear-decay':
        epsilons = get_epsilons_linear_decay(
            n_episodes=kwargs['n_episodes'],
            eps_min=kwargs['eps_min'],
        )

    # exponential decaying sequence
    elif eps_type == 'exp-decay':
        epsilons = get_epsilons_exp_decay(
            n_episodes=kwargs['n_episodes'],
            eps_init=kwargs['eps_init'],
            eps_decay=kwargs['eps_decay']
        )

    # harmonic decreasing sequence
    elif eps_type == 'harmonic':
        epsilons = get_epsilons_harmonic(
            n_episodes=kwargs['n_episodes'],
        )

    return epsilons

def get_epsilons_constant(n_episodes, eps_init):
    return eps_init * np.ones(n_episodes)

def get_epsilons_linear_decay(n_episodes, eps_min, exploration=0.75):
    n_episodes_exploration = int(n_episodes * exploration)
    return np.array([
            1 + (eps_min - 1) * min(1, ep / n_episodes_exploration)
            for ep in range(n_episodes)
    ])

def get_epsilons_exp_decay(n_epsisodes, eps_init, eps_decay):
    self.epsilons = np.array([
        #self.eps_min + (self.eps_max - self.eps_min) * 10**(-self.eps_decay * ep)
        eps_init * (eps_decay ** ep)
        for ep in np.arange(n_episodes)
    ])

def get_epsilons_harmonic(n_episodes):
    return np.array([1 / (ep + 1) for ep in np.arange(n_episodes)])


def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def compute_tables(env, q_table):

    # compute value function
    v_table = np.max(q_table, axis=1)

    # compute advantage table
    a_table = q_table - np.expand_dims(v_table, axis=1)

    # compute greedy actions
    greedy_policy = env.get_greedy_actions(q_table)
    greedy_policy[env.idx_lb:] = env.idx_null_action

    return v_table, a_table, greedy_policy

def initialize_figures(env, n_table, q_table, n_episodes, value_function_hjb, control_hjb):

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]
    ax5, ax6 = axes[:, 2]

    # frequency subplot
    ax1.set_title('Frequency table')
    #ax1.set_xlabel('Discretized state space')
    #ax1.set_ylabel('Discretized action space')

    ax2.set_title('Q table')
    #ax2.set_xlabel('Discretized state space')
    #ax2.set_ylabel('Discretized action space')

    ax3.set_title('Value function')
    #ax3.set_xlabel('Discretized state space')
    ax3.set_xlim(env.state_space_low, env.state_space_high)
    ax3.set_ylim(-3, 1)

    ax4.set_title('Greedy policy')
    #ax4.set_xlabel('Discretized state space')
    #ax4.set_ylabel('Discretized action space')
    ax4.set_xlim(env.state_space_low, env.state_space_high)
    ax4.set_ylim(env.action_space_low, env.action_space_high)

    ax5.set_title('Return')
    ax5.set_xlim(0, n_episodes)
    ax5.set_ylim(-10, 1)

    ax6.set_title('Time steps')
    ax6.set_xlim(0, n_episodes)
    ax6.set_ylim(0, 1000)

    plt.ion()

    extent = env.state_space_low - env.h_state / 2, env.state_space_high + env.h_state / 2, \
             env.action_space_low - env.h_action / 2, env.action_space_high + env.h_action / 2

    # n table
    im1 = ax1.imshow(
        n_table.T,
        vmin=0,
        vmax=10**4,
        extent=extent,
        origin='lower',
        cmap=cm.coolwarm,
        aspect='auto',
    )

    # q table
    im2 = ax2.imshow(
        q_table.T,
        origin='lower',
        extent=extent,
        cmap=cm.viridis,
        aspect='auto',
    )

    # value function
    v_table = np.max(q_table, axis=1)
    value_function_line = ax3.plot(env.state_space_h, v_table)[0]
    ax3.plot(env.state_space_h, -value_function_hjb)

    # control
    greedy_actions = env.get_greedy_actions(q_table)
    policy_line = ax4.plot(env.state_space_h, greedy_actions)[0]
    ax4.plot(env.state_space_h, control_hjb)

    # episodes
    episodes = np.arange(n_episodes)

    # returns
    returns = np.empty_like(episodes, dtype=np.float32)
    avg_returns = np.empty_like(episodes, dtype=np.float32)
    returns.fill(np.nan)
    avg_returns.fill(np.nan)
    returns_line = ax5.plot(episodes, returns)[0]
    avg_returns_line = ax5.plot(episodes, avg_returns)[0]

    # time steps
    time_steps = np.empty_like(episodes, dtype=np.float32)
    avg_time_steps = np.empty_like(episodes, dtype=np.float32)
    time_steps.fill(np.nan)
    avg_time_steps.fill(np.nan)
    time_steps_line = ax6.plot(episodes, time_steps)[0]
    avg_time_steps_line = ax6.plot(episodes, avg_time_steps)[0]

    plt.show()
    images = (im1, im2)
    lines = (value_function_line, policy_line, returns_line, avg_returns_line,
             time_steps_line, avg_time_steps_line)
    return images, lines

def update_figures(env, n_table, q_table, returns, avg_returns,
                   time_steps, avg_time_steps, images, lines):

    # update n and q table
    im1, im2 = images
    im1.set_data(n_table.T)
    im2.set_data(q_table.T)

    # compute value function and greedy actions
    v_table = np.max(q_table, axis=1)
    greedy_actions = env.get_greedy_actions(q_table)

    # episodes
    episodes = np.arange(returns.shape[0])

    value_function_line, policy_line, returns_line, avg_returns_line, \
        time_steps_line, avg_time_steps_line = lines

    # update plots
    value_function_line.set_data(env.state_space_h, v_table)
    policy_line.set_data(env.state_space_h, greedy_actions)
    returns_line.set_data(episodes, returns)
    avg_returns_line.set_data(episodes, avg_returns)
    time_steps_line.set_data(episodes, time_steps)
    avg_time_steps_line.set_data(episodes, avg_time_steps)

    # update figure frequency
    plt.pause(0.01)
