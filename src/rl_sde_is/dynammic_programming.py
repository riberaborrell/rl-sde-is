import numpy as np
import matplotlib.pyplot as plt

def compute_p_tensor_batch(env):

    # initialize state action transition tensor
    p_tensor = np.empty((env.n_states, env.n_states, env.n_actions))
    for idx_state in range(env.n_states):
        for idx_action in range(env.n_actions):
            state = env.state_space_h[idx_state]
            action = env.action_space_h[idx_action]
            p_tensor[:, idx_state, idx_action] \
                = env.state_action_transition_function(env.state_space_h, state, action, env.h_state /2 )

    # set values for the target set and null action
    p_tensor[env.idx_lb:, env.idx_lb:, :] = 1
    p_tensor[:env.idx_lb, env.idx_lb:, :] = 0

    return p_tensor

def compute_r_table(env):

    # initialize expected reward table
    r_table = np.empty((env.n_states, env.n_actions))

    for idx_action in range(env.n_actions):
        action = env.action_space_h[idx_action]
        r_table[:, idx_action] = env.reward_signal(env.state_space_h, action)

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

def plot_policy(env, policy, control_hjb):

    # discretize state space
    x = env.state_space_h

    # get actions following given policy
    actions = np.array([
        env.action_space_h[idx_action] for idx_action in policy
    ])

    fig, ax = plt.subplots()
    ax.set_title('Policy')
    ax.plot(x, actions, label='dp')
    ax.plot(x, control_hjb[:, 0], label='hjb solution')
    ax.legend()
    plt.show()

def plot_value_function(env, v_table, value_f_hjb):

    # discretize state space
    x = env.state_space_h

    fig, ax = plt.subplots()
    ax.set_title('Value function')
    ax.plot(x, -v_table, label='dp')
    ax.plot(x, value_f_hjb, label='hjb solution')
    #fig.set_ylim(-100, 0)
    ax.legend()
    plt.show()
