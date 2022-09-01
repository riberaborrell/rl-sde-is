import numpy as np

from base_parser import get_base_parser
from dynammic_programming import compute_p_tensor_batch, compute_r_table, \
                                 plot_policy, plot_value_function
from environments import DoubleWellStoppingTime1D

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def value_iteration(env, gamma=1.0, n_iterations=100, n_avg_iterations=10):
    ''' value iteration with synchronous back ups
    '''
    # compute p tensor and r table
    p_tensor = compute_p_tensor_batch(env)
    r_table = compute_r_table(env)

    # initialize value function table
    v_table = np.zeros(env.n_states)

    # set values for the target set
    v_table[env.idx_lb:env.idx_rb+1] = 0

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init)

    # value iteration
    for i in np.arange(n_iterations):

        # copy value function table
        v_table_i = v_table.copy()

        # loop over states not in the target set
        for idx_state in range(env.idx_lb):

            # preallocate values
            values = np.zeros(env.n_actions)

            # loop over all actions
            for idx_action in range(env.n_actions):

                values[idx_action] = r_table[idx_state, idx_action] \
                                   + gamma \
                                   * np.dot(
                                       p_tensor[np.arange(env.n_states), idx_state, idx_action],
                                       v_table_i[np.arange(env.n_states)],
                                    )

            # Bellman optimality equation
            v_table[idx_state] = np.max(values)


        # logs
        if i % n_avg_iterations == 0:
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, v_table[idx_state_init])
            print(msg)

    # initialize policy
    env.get_idx_null_action()
    policy = np.empty(env.n_states, dtype=np.int32)
    policy[env.idx_lb:] = env.idx_null_action

    # policy computation.
    for idx_state in range(env.idx_lb):
        # preallocate values
        values = np.zeros(env.n_actions)
        for idx_action in range(env.n_actions):

            values[idx_action] = r_table[idx_state, idx_action] \
                               + gamma * np.dot(
                                       p_tensor[np.arange(env.n_states), idx_state, idx_action],
                                       v_table_i[np.arange(env.n_states)],
                                    )
            # Bellman optimality equation
            policy[idx_state] = np.argmax(values)

    return v_table, policy

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get target set indices
    env.get_idx_target_set()

    # run dynamic programming value iteration
    v_table, policy = value_iteration(
        env,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        n_avg_iterations=args.n_avg_iterations,
    )

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # do plots
    plot_value_function(env, v_table, value_f_hjb=sol_hjb.value_function)
    plot_policy(env, policy, control_hjb=sol_hjb.u_opt)


if __name__ == '__main__':
    main()
