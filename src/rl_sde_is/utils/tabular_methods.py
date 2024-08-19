import numpy as np
import matplotlib.pyplot as plt

def compute_value_function(q_table):
    return np.max(q_table, axis=1)

def compute_advantage_function(v_table, q_table):
    return q_table - np.expand_dims(v_table, axis=1)

def compute_greedy_action_indices(q_table):
    return np.argmax(q_table, axis=1)

def compute_greedy_policy(env, q_table):
    actions_idx = compute_greedy_action_indices(q_table)
    return env.action_space_h[actions_idx]

def compute_value_advantage_and_greedy_actions(q_table):
    ''' computes the value table, the advantage table and the greedy action indices.
    '''
    v_table = compute_value_function(q_table)
    a_table = compute_advantage_function(v_table, q_table)
    actions_idx = compute_greedy_action_indices(q_table)

    return v_table, a_table, actions_idx

def compute_value_advantage_and_greedy_policy(env, q_table):
    ''' computes the value table, the advantage table and the greedy policy.
    '''

    v_table = compute_value_function(q_table)
    a_table = compute_advantage_function(v_table, q_table)
    actions_idx = compute_greedy_action_indices(q_table)
    greedy_policy = env.action_space_h[actions_idx]
    greedy_policy[env.target_set_idx] = 0

    return v_table, a_table, greedy_policy

def get_det_policy_indices(env, policy):
    return env.get_action_idx(policy)

def compute_rms_error(table, appr_table):

    # get number of states
    n_states = table.shape[0]

    return np.linalg.norm(table - appr_table) / np.sqrt(n_states)
    # return np.linalg.norm(table - appr_table)

def get_epsilon_greedy_action(env, q_table, state_idx, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        action_idx = np.argmax(q_table[state_idx])

    # pick random action (exploration)
    else:
        action_idx = np.random.choice(np.arange(env.n_actions))

    action = env.action_space_h[action_idx]
    return action_idx, action

def get_epsilon_greedy_actions_vect(env, q_table, states_idx, epsilon):

    # get batch size
    batch_size = states_idx.shape[0]

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        actions_idx = np.argmax(q_table[states_idx], axis=1)

    # pick random action (exploration)
    else:
        actions_idx = np.random.choice(np.arange(env.n_actions), batch_size)

    actions = env.action_space_h[actions_idx].reshape(batch_size, 1)

    return actions_idx, actions

def get_epsilons_constant(n_episodes, eps_init):
    return eps_init * np.ones(n_episodes)

def get_epsilons_linear_decay(n_episodes, eps_min, exploration=0.75):
    n_episodes_exploration = int(n_episodes * exploration)
    return np.array([
            1 + (eps_min - 1) * min(1, ep / n_episodes_exploration)
            for ep in range(n_episodes)
    ])

def get_epsilons_exp_decay(n_episodes, eps_init, eps_decay):
    return np.array([
        #self.eps_min + (self.eps_max - self.eps_min) * 10**(-self.eps_decay * ep)
        eps_init * (eps_decay ** ep)
        for ep in np.arange(n_episodes)
    ])

def get_epsilons_harmonic(n_episodes):
    return np.array([1 / (ep + 1) for ep in np.arange(n_episodes)])
