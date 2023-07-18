import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from tabular_methods import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def value_table_update(env, r_table, p_tensor, policy, v_table, gamma):

    # copy value function table
    v_table_i = v_table.copy()

    # loop over state indices
    for state_idx in range(env.n_states):

        # check if is in target set
        d = 1 if  state_idx in env.idx_ts else 0

        # choose action following policy
        action_idx = policy[state_idx]

        # update v table
        v_table[state_idx] = r_table[state_idx, action_idx]

        # loop over next state indices
        for next_state_idx in range(env.n_states):

            # update v table
            v_table[state_idx] += (1 - d) * gamma \
                               * p_tensor[next_state_idx, state_idx, action_idx] \
                               * v_table_i[next_state_idx]

def value_table_update_semi_vect(env, r_table, p_tensor, policy, v_table, gamma):
    v_table_i = v_table.copy()
    d = np.where(env.is_in_ts, 1, 0)
    for state_idx in range(env.n_states):
        action_idx = policy[state_idx]
        v_table[state_idx] = r_table[state_idx, action_idx] \
                           + gamma * (1 - d[state_idx]) * np.dot(
                                p_tensor[:, state_idx, action_idx].squeeze(),
                                v_table_i,
                           )

def value_table_update_vect(env, r_table, p_tensor, policy, v_table, gamma):
    actions_idx = policy.squeeze()
    d = np.where(env.is_in_ts, 1, 0)
    v_table = r_table[np.arange(env.n_states), actions_idx] \
            + gamma * (1 - d) * np.matmul(
                p_tensor[:, np.arange(env.n_states), actions_idx].T,
                v_table
            )
    return v_table

def policy_evaluation(env, gamma=1.0, n_iterations=100, test_freq_iterations=10,
                      policy=None, value_function_opt=None, load=False, live_plot=False):

    ''' Dynamic programming policy evaluation.
    '''
    # get dir path
    rel_dir_path = get_dynamic_programming_dir_path(
        env,
        agent='dp-prediction',
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
    v_table = - np.random.rand(env.n_states)

    # get index initial state
    state_init_idx = env.get_state_idx(env.state_init).item()

    # preallocate value function rms errors
    n_test_iterations = n_iterations // test_freq_iterations + 1
    v_rms_errors = np.empty(n_test_iterations)

    # compute errors
    v_rms_errors[0] = compute_rms_error(value_function_opt, v_table)

    # initialize live figures
    if live_plot:
        line = initialize_value_function_1d_figure(env, v_table, value_function_opt)

    # for each iteration
    for i in np.arange(n_iterations):

        #value_table_update(env, r_table, p_tensor, policy, v_table, gamma)
        #value_table_update_semi_vect(env, r_table, p_tensor, policy, v_table, gamma)
        v_table = value_table_update_vect(env, r_table, p_tensor, policy, v_table, gamma)

        # test
        if (i + 1) % test_freq_iterations == 0:

            # compute root mean square error of value function
            j = (i + 1) // test_freq_iterations
            v_rms_errors[j] = compute_rms_error(value_function_opt, v_table)

            # logs
            msg = 'it: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}' \
                  ''.format(i+1, v_table[state_init_idx], v_rms_errors[j])
            print(msg)

            # update live figure
            if live_plot:
                update_value_function_1d_figure(env, v_table, line)

    data = {
        'n_iterations': n_iterations,
        'v_table' : v_table,
        'v_rms_errors' : v_rms_errors,
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

    # set deterministic policy from the hjb control
    policy = np.array([
        env.get_action_idx(sol_hjb.u_opt[state_idx])
        for state_idx, _ in enumerate(env.state_space_h)
    ])

    # run dynammic programming policy evaluation of the optimal policy
    data = policy_evaluation(
        env,
        policy=policy,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        test_freq_iterations=args.test_freq_iterations,
        value_function_opt=-sol_hjb.value_function,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plot
    if not args.plot:
        return

    # do plots
    policy = env.action_space_h[policy]
    plot_det_policy_1d(env, policy, sol_hjb.u_opt)
    plot_value_function_1d(env, data['v_table'], -sol_hjb.value_function)
    plot_value_rms_error_iterations(data['v_rms_errors'], args.test_freq_iterations)

if __name__ == '__main__':
    main()
