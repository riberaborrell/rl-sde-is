import numpy as np

def compute_r_table(env):

    # initialize expected reward table
    r_table = np.empty((env.n_states, env.n_actions))

    # all discretized states
    states = np.expand_dims(env.state_space_h, axis=-1)
    done = env.is_done(states)

    for action_idx in range(env.n_actions):
        action = env.action_space_h[[action_idx]].reshape(1, env.d)
        r_table[:, action_idx] = env.reward_signal_state_action(states, action, done)

    return r_table

def compute_p_tensor_batch(env):

    # initialize state action transition tensor
    p_tensor = np.empty((env.n_states, env.n_states, env.n_actions))
    p_tensor[:, :, :] = np.nan

    # set values for the state in target set
    p_tensor[env.idx_ts[:, np.newaxis], env.idx_ts, :] = 1 / env.is_in_ts.sum()
    p_tensor[env.idx_not_ts[:, np.newaxis], env.idx_ts, :] = 0

    # compute rest of state action probabilities
    for state_idx in env.idx_not_ts:
        for action_idx in range(env.n_actions):
            state = env.state_space_h[state_idx]
            action = env.action_space_h[action_idx]
            p_tensor[:, state_idx, action_idx] \
                = env.state_action_transition_function(env.state_space_h, state, action, env.h_state /2 )

    return p_tensor
