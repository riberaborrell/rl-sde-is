import numpy as np

from base_parser import get_base_parser
from dynammic_programming import compute_p_tensor_batch, compute_r_table, \
                                 plot_policy, plot_value_function
from environments import DoubleWellStoppingTime1D

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def qvalue_iteration(env, gamma=1.0, n_iterations=100, n_avg_iterations=10, n_steps_lim=10000):
    '''
    '''
    # compute p tensor and r table
    p_tensor = compute_p_tensor_batch(env)
    r_table = compute_r_table(env)

    # initialize value function table
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set and null action
    q_table[env.idx_lb:env.idx_rb+1] = 0

    # get index x_init
    idx_state_init = env.get_state_idx(env.state_init)

    # for each iteration
    for i in np.arange(n_iterations):

        # reset environment
        state = env.state_init.copy()

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # get index of the state
            idx_state = env.get_state_idx(state)

            # choose greedy action
            idx_action = np.argmax(q_table[idx_state])
            action = env.action_space_h[[idx_action]]

            value = r_table[idx_state, idx_action] \
                  + gamma \
                  * np.dot(
                      p_tensor[np.arange(env.n_states), idx_state, idx_action],
                      np.max(q_table, axis=1),
                  )

            q_table[idx_state, idx_action] = value

            # step dynamics forward
            state, r, complete = env.step(state, action)

        # logs
        if i % n_avg_iterations == 0:
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, np.max(q_table[idx_state_init]))
            print(msg)

    return q_table


def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # discretize state and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get target set indices
    env.get_idx_target_set()

    # run dynamic programming tabular method 
    q_table = qvalue_iteration(
        env,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        n_avg_iterations=args.n_avg_iterations,
    )

    # idx target set
    idx_null_action = env.get_action_idx(0.)

    # compute optimal policy and value function
    policy = np.argmax(q_table, axis=1)
    policy[env.idx_lb:] = idx_null_action
    v_table = np.max(q_table, axis=1)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # factor between the two different discretizations steps
    k = int(args.h_state / sol_hjb.sde.h)
    assert env.state_space_h.shape == sol_hjb.u_opt[::k, 0].shape, ''

    # do plots
    plot_policy(env, policy, control_hjb=sol_hjb.u_opt[::k])
    plot_value_function(env, v_table, value_f_hjb=sol_hjb.value_function[::k])


if __name__ == '__main__':
    main()
