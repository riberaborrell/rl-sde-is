import numpy as np

def compute_p_tensor_batch(env):

    # initialize state action transition tensor
    p_tensor = np.empty((env.n_states, env.n_states, env.n_actions))

    # set values for the state in target set
    p_tensor[env.idx_ts[:, np.newaxis], env.idx_ts, :] = 1
    p_tensor[env.idx_not_ts[:, np.newaxis], env.idx_ts, :] = 0

    # compute rest of state action probabilities
    for idx_state in env.idx_not_ts:
        for idx_action in range(env.n_actions):
            state = env.state_space_h[idx_state]
            action = env.action_space_h[idx_action]
            p_tensor[:, idx_state, idx_action] \
                = env.state_action_transition_function(env.state_space_h, state, action, env.h_state /2 )

    return p_tensor

def compute_r_table(env):

    # initialize expected reward table
    r_table = np.empty((env.n_states, env.n_actions))

    # all discretized states
    states = np.expand_dims(env.state_space_h, axis=-1)

    for idx_action in range(env.n_actions):
        action = env.action_space_h[[idx_action]].reshape(1, env.d)
        r_table[:, idx_action], _ = env.reward_signal_state_action(states, action)

    return r_table

def check_bellman_equation():
    pass
    # check that p_tensor and r_table are well computed
    #a = -hjb_value_f[idx_x_init]
    #b = np.dot(
    #   p_tensor[np.arange(env.n_states), idx_x_init, policy[idx_x_init]],
    #   - hjb_value_f[np.arange(env.n_states)],
    #) + rew
    #assert a == b, ''
