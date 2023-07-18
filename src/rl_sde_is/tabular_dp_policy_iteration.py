import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import *
from rl_sde_is.tabular_dp_value_iteration import v_table_update_vect
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def policy_update_semi_vect(env, r_table, p_tensor, v_table, policy_indices, gamma):

    d = np.where(env.is_in_ts, 1, 0)

    # loop over states not in the target set
    for state_idx in range(env.idx_lb):

        # preallocate values
        values = np.zeros(env.n_actions)

        # loop over all actions
        for action_idx in range(env.n_actions):

            values[action_idx] = r_table[state_idx, action_idx] \
                               + gamma * np.dot(
                                   p_tensor[np.arange(env.n_states), state_idx, action_idx],
                                   v_table[np.arange(env.n_states)],
                               )

        # Bellman optimality equation
        policy_indices[state_idx] = np.argmax(values)

def policy_update_vect(env, r_table, p_tensor, v_table, gamma):

    d = np.where(env.is_in_ts, 1, 0)[:, None]
    values = r_table \
           + (1 - d) * gamma * np.matmul(
                np.swapaxes(p_tensor, 0, 2),
                v_table,
            ).transpose()

    # Bellman optimality equation
    policy_indices = np.argmax(values, axis=1)
    policy_indices[env.idx_ts] = env.idx_null_action
    return policy_indices



def policy_iteration(env, gamma=1.0, n_iterations=100, test_freq_iterations=10,
                     value_function_opt=None, policy_opt=None, load=False, live_plot=False):

    ''' Dynamic programming policy iteration.
    '''
    # get dir path
    rel_dir_path = get_dynamic_programming_dir_path(
        env,
        agent='dp-policy-iteration',
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

    # initialize value function table and policy (array storing the indices of the actions)
    v_table = - np.random.rand(env.n_states)
    policy_indices = np.random.randint(env.n_actions, size=env.n_states)
    policy = env.action_space_h[policy_indices]

    # set values for the target set
    policy_indices[env.idx_ts] = env.idx_null_action

    # get index initial state
    state_idx_init = env.get_state_idx(env.state_init).item()

    # preallocate value function rms errors
    n_test_iterations = n_iterations // test_freq_iterations + 1
    v_rms_errors = np.empty(n_test_iterations)
    p_rms_errors = np.empty(n_test_iterations)

    # compute errors
    v_rms_errors[0] = compute_rms_error(value_function_opt, v_table)
    p_rms_errors[0] = compute_rms_error(policy_opt, policy)

    if live_plot:
        v_line = initialize_value_function_1d_figure(env, v_table, value_function_opt)
        p_line = initialize_det_policy_1d_figure(env, policy, policy_opt)

    # in each value iteration and policy update
    for i in np.arange(n_iterations):

        # value function update
        v_table = v_table_update_vect(env, r_table, p_tensor, v_table, gamma)

        # policy update
        #policy_update_semi_vect(env, r_table, p_tensor, v_table, policy_indices, gamma)
        policy_indices = policy_update_vect(env, r_table, p_tensor, v_table, gamma)


        # test
        if (i + 1) % test_freq_iterations == 0:

            # update policy
            policy = env.action_space_h[policy_indices]

            # compute root mean square error of value function and policy
            j = (i + 1) // test_freq_iterations
            v_rms_errors[j] = compute_rms_error(value_function_opt, v_table)
            p_rms_errors[j] = compute_rms_error(policy_opt, policy)

            # logs
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, v_table[state_idx_init])
            print(msg)

            # update live figures
            if live_plot:
                update_value_function_1d_figure(env, v_table, v_line)
                update_det_policy_1d_figure(env, policy, p_line)

    # compute policy actions
    policy = env.action_space_h[policy_indices]

    data = {
        'n_iterations': n_iterations,
        'v_table' : v_table,
        'policy' : policy,
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

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run dynamic programming policy iteration
    data = policy_iteration(
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

    # do plots
    plot_value_function_1d(env, data['v_table'], -sol_hjb.value_function)
    plot_det_policy_1d(env, data['policy'], sol_hjb.u_opt)
    plot_value_rms_error_iterations(data['v_rms_errors'], args.test_freq_iterations)
    plot_policy_rms_error_iterations(data['p_rms_errors'], args.test_freq_iterations)

if __name__ == '__main__':
    main()
