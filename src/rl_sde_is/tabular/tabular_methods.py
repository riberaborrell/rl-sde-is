import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_epsilon_greedy_action(env, q_table, state_idx, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        action_idx = np.argmax(q_table[state_idx])

    # pick random action (exploration)
    else:
        action_idx = np.random.choice(np.arange(env.n_actions))

    action = env.action_space_h[[[action_idx]]]

    return action_idx, action

def get_epsilon_greedy_actions_vectorized(env, q_table, states_idx, epsilon):

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

def compute_value_function(q_table):
    return np.max(q_table, axis=1)

def compute_value_advantage_and_greedy_actions(q_table):
    ''' computes the value table, the advantage table and the greedy action indices.
    '''
    # compute value function
    v_table = compute_value_function(q_table)

    # compute advantage table
    a_table = q_table - np.expand_dims(v_table, axis=1)

    # compute greedy action indices
    actions_idx = np.argmax(q_table, axis=1)

    return v_table, a_table, actions_idx

def compute_tables(env, q_table):
    ''' computes the value table, the advantage table and the greedy action indices.
    '''

    # compute value function
    v_table = np.max(q_table, axis=1)

    # compute advantage table
    a_table = q_table - np.expand_dims(v_table, axis=1)

    # compute greedy actions
    greedy_policy = env.get_greedy_actions(q_table)
    greedy_policy[env.ts_idx] = 0

    return v_table, a_table, greedy_policy

def compute_rms_error(table, appr_table):

    # get number of states
    n_states = table.shape[0]

    return np.linalg.norm(table - appr_table) / np.sqrt(n_states)
    # return np.linalg.norm(table - appr_table)

def evaluate_policy_vectorized(env, policy, batch_size=10, k_max=10**7):

    # preallocate returns and time steps
    cum_rewards = np.zeros(batch_size)
    returns = np.empty(batch_size)
    time_steps = np.empty(batch_size)

    # are episodes done
    already_done = np.full((batch_size,), False)
    done = np.full((batch_size,), False)

    # initialize episodes
    states = np.full((batch_size, env.d), env.state_init)

    # sample episodes
    for k in np.arange(k_max):

        # get index of the states
        states_idx = env.get_state_idx(states)

        # actions
        actions = np.expand_dims(env.action_space_h[policy[states_idx]], axis=1)

        # step dynamics forward
        next_states, rewards, done, dbt = env.step(states, actions)

        # update cumulative rewards for all trajectories
        cum_rewards += np.squeeze(rewards)

        # get indices of episodes which are new to the target set
        idx = env.get_new_in_ts_idx(done, already_done)

        # if there are episodes which are done
        if idx.shape[0] != 0:

            # fix episode returns
            returns[idx] = cum_rewards[idx]

            # fix episode time steps
            time_steps[idx] = k

        # stop if xt_traj in target set
        if already_done.all() == True:
           break

        # update states
        states = next_states

    if not already_done.all():
        return np.nan, np.nan
    else:
        return returns, time_steps
