import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def q_table_update_semi_vect(env, r_table, p_tensor, q_table, gamma):

    # copy value function table
    q_table_i = q_table.copy()

    d = np.where(env.is_in_ts, 1, 0)

    # loop over states not in the target set 
    for state_idx in range(env.n_states):

        # loop over all actions
        for action_idx in range(env.n_actions):
            value = r_table[state_idx, action_idx] \
                   + (1 - d[state_idx]) * gamma \
                   * np.dot(
                        p_tensor[np.arange(env.n_states), state_idx, action_idx],
                        np.max(q_table_i, axis=1),
                    )

            q_table[state_idx, action_idx] = value

def q_table_update_vect(env, r_table, p_tensor, q_table, gamma):

    d = np.where(env.is_in_ts, 1, 0)[:, None]
    q_table = r_table \
            + (1 - d) * gamma * np.matmul(
                np.swapaxes(p_tensor, 0, 2), np.max(q_table, axis=1)
            ).transpose()

    return q_table

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

    # load dp tables
    tables_data = load_data(get_dynamic_programming_tables_dir_path(env))
    r_table = tables_data['r_table']
    p_tensor = tables_data['p_tensor']

    # initialize value function table
    q_table = - np.random.rand(env.n_states, env.n_actions)

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
        lines = initialize_tabular_figures(env, q_table, v_table, a_table, policy,
                                           value_function_opt, policy_opt)
    # for each iteration
    for i in np.arange(n_iterations):


        #q_table_update_semi_vect(env, r_table, p_tensor, q_table, gamma)
        q_table = q_table_update_vect(env, r_table, p_tensor, q_table, gamma)

        # test
        if (i + 1) % test_freq_iterations == 0:

            # compute tables
            v_table, a_table, policy = compute_tables(env, q_table)

            # compute root mean square error of value function and policy
            j = (i + 1) // test_freq_iterations
            v_rms_errors[j] = compute_rms_error(value_function_opt, v_table)
            p_rms_errors[j] = compute_rms_error(policy_opt, policy)

            # logs
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i+1, np.max(q_table[env.state_init_idx]))
            print(msg)

            # update live figures
            if live_plot:
                update_tabular_figures(env, q_table, v_table, a_table, policy, lines)

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

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # set action space bounds
    env.set_action_space_bounds()

    # discretize state and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run dynamic programming q-value iteration
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
