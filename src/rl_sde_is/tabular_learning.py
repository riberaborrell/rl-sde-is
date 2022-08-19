import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_epsilon_greedy_action(env, q_table, epsilon, idx_state):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        idx_action = np.argmax(q_table[idx_state])

    # pick random action (exploration)
    else:
        idx_action = np.random.choice(np.arange(env.n_actions))

    action = env.action_space_h[[idx_action]]

    return idx_action, action

def get_epsilons_constant(n_episodes, eps_init):
    return eps_init * np.ones(n_episodes)

def get_epsilons_harmonic(n_episodes):
    return np.array([1 / (ep + 1) for ep in np.arange(n_episodes)])

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

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def plot_frequency_table(env, n_table):
    # set extent bounds
    extent = env.state_space_h[0], env.state_space_h[-1], \
             env.action_space_h[0], env.action_space_h[-1]

    fig, ax = plt.subplots()

    im = fig.axes[0].imshow(
        n_table.T,
        origin='lower',
        extent=extent,
        cmap=cm.coolwarm,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_q_table(env, q_table):
    # set extent bounds
    extent = env.state_space_h[0], env.state_space_h[-1], \
             env.action_space_h[0], env.action_space_h[-1]

    fig, ax = plt.subplots()

    im = fig.axes[0].imshow(
        q_table.T,
        origin='lower',
        extent=extent,
        cmap=cm.viridis,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_v_table(env, q_table, value_function_hjb):

    x = env.state_space_h
    v_table = np.max(q_table, axis=1)

    fig, ax = plt.subplots()
    plt.plot(x, -v_table)
    plt.plot(x, value_function_hjb)
    plt.show()


def plot_greedy_policy(env, q_table, control_hjb):

    x = env.state_space_h

    # compute greedy policy by following the q-table
    greedy_policy = np.empty_like(x)
    for idx, x_k in enumerate(x):
        idx_action = np.argmax(q_table[idx])
        greedy_policy[idx] = env.action_space_h[idx_action]

    fig, ax = plt.subplots()
    plt.plot(x, greedy_policy)
    plt.plot(x, control_hjb[:, 0])
    plt.show()
