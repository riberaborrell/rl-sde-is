import numpy as np

def compute_r_table(env):

    # initialize expected reward table
    r_table = np.empty((env.n_states, env.n_actions))

    # all discretized states
    states = env.state_space_h.flatten()
    done = env.is_target_set(states)

    for action_idx in range(env.n_actions):
        action = env.action_space_h[action_idx]
        action = np.expand_dims(action, axis=-1)
        r_table[:, action_idx] = env.reward_state_action_vect_fn(states, action, done)

    return r_table

def compute_p_tensor_batch(env):

    # initialize state action transition tensor
    p_tensor = np.empty((env.n_states, env.n_states, env.n_actions))
    p_tensor[:, :, :] = np.nan

    # set values for the state in target set
    p_tensor[env.target_set_idx[:, np.newaxis], env.target_set_idx, :] = 1 / env.target_set_idx.shape[0]
    p_tensor[env.not_target_set_idx[:, np.newaxis], env.target_set_idx, :] = 0

    # compute rest of state action probabilities
    for state_idx in env.not_target_set_idx:
        for action_idx in range(env.n_actions):
            state = env.state_space_h[state_idx]
            action = env.action_space_h[action_idx]
            p_tensor[:, state_idx, action_idx] = env.state_action_transition_function_1d(
                env.state_space_h, state, action, env.h_state /2
            ).squeeze()

    return p_tensor
