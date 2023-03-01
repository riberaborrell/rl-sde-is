import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.dynammic_programming import compute_p_tensor_batch, compute_r_table
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def qvalue_iteration(env, gamma=1.0, n_iterations=100, test_freq_iterations=10,
                     value_function_opt=None, policy_opt=None, load=False, live_plot=False):

    ''' Dynamic programming q-value iteration.
    '''

    # get dir path
    rel_dir_path = get_dynamic_programming_dir_path(
        env,
        agent='dp-q-value-iteration',
        n_iterations=n_iterations,
    )

    # load results
    if load:
        data = load_data(rel_dir_path)
        return data

    # compute p tensor and r table
    p_tensor = compute_p_tensor_batch(env)
    r_table = compute_r_table(env)

    # initialize value function table
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set and null action
    q_table[env.idx_ts] = 0

    # get index x_init
    idx_state_init = env.get_state_idx(env.state_init).item()

    # preallocate value function rms errors
    n_test_iterations = n_iterations // test_freq_iterations + 1
    v_rms_errors = np.empty(n_test_iterations)
    p_rms_errors = np.empty(n_test_iterations)

    # compute tables
    v_table, a_table, policy = compute_tables(env, q_table)

    # compute errors
    v_rms_errors[0] = compute_rms_error(value_function_opt, v_table)
    p_rms_errors[0] = compute_rms_error(policy_opt, policy)

    # initialize live figures
    if live_plot:
        lines = initialize_q_learning_figures(env, q_table, v_table, a_table, policy,
                                              value_function_opt, policy_opt)
    # for each iteration
    for i in np.arange(n_iterations):

        # copy value function table
        q_table_i = q_table.copy()

        # loop over states not in the target set 
        for idx_state in env.idx_not_ts:

            # loop over all actions
            for idx_action in range(env.n_actions):
                value = r_table[idx_state, idx_action] \
                      + gamma \
                      * np.dot(
                          p_tensor[np.arange(env.n_states), idx_state, idx_action],
                          np.max(q_table_i, axis=1),
                        )

                q_table[idx_state, idx_action] = value

        # test
        if (i + 1) % test_freq_iterations == 0:

            # compute tables
            v_table, a_table, policy = compute_tables(env, q_table)

            # compute root mean square error of value function and policy
            j = (i + 1) // test_freq_iterations
            v_rms_errors[j] = compute_rms_error(value_function_opt, v_table)
            p_rms_errors[j] = compute_rms_error(policy_opt, policy)

            # logs
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, np.max(q_table[idx_state_init]))
            print(msg)

            # update live figures
            if live_plot:
                update_q_learning_figures(env, q_table, v_table, a_table, policy, lines)

    data = {
        'n_iterations': n_iterations,
        'q_table' : q_table,
        'v_rms_errors' : v_rms_errors,
        'p_rms_errors' : p_rms_errors,
    }
    save_data(data, rel_dir_path)
    return data


def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # discretize state and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run dynamic programming tabular method 
    data = qvalue_iteration(
        env,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        test_freq_iterations=args.test_freq_iterations,
        value_function_opt=-sol_hjb.value_function,
        policy_opt=sol_hjb.u_opt,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plot
    if not args.plot:
        return

    # compute tables
    q_table = data['q_table']
    v_table, a_table, policy_greedy = compute_tables(env, q_table)

    # do plots
    plot_value_function_1d(env, v_table, -sol_hjb.value_function)
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table)
    plot_det_policy_1d(env, policy_greedy, sol_hjb.u_opt)
    plot_value_rms_error_iterations(data['v_rms_errors'], args.test_freq_iterations)
    plot_policy_rms_error_iterations(data['p_rms_errors'], args.test_freq_iterations)

if __name__ == '__main__':
    main()
