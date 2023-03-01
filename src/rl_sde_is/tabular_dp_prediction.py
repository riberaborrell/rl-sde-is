import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.dynammic_programming import compute_p_tensor_batch, compute_r_table
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from tabular_methods import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

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

    # compute p tensor and r table
    p_tensor = compute_p_tensor_batch(env)
    r_table = compute_r_table(env)

    # initialize value function table
    v_table = - np.random.rand(env.n_states)

    # set values for the target set
    v_table[env.idx_ts] = 0

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init).item()

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

        # copy value function table
        v_table_i = v_table.copy()

        for idx_state in range(env.idx_lb):
            idx_action = policy[idx_state]

            #v_table[idx_state] = r_table[idx_state, idx_action]
            #for idx_next_state in range(env.n_states):
            #    v_table[idx_state] += gamma \
            #                       * p_tensor[idx_next_state, idx_state, idx_action] \
            #                       * v_table[idx_next_state]

            v_table[idx_state] = r_table[idx_state, idx_action] \
                               + gamma * np.dot(
                                   p_tensor[np.arange(env.n_states), idx_state, idx_action],
                                   v_table_i[np.arange(env.n_states)],
                               )
        # test
        if (i + 1) % test_freq_iterations == 0:

            # compute root mean square error of value function
            j = (i + 1) // test_freq_iterations
            v_rms_errors[j] = compute_rms_error(value_function_opt, v_table)

            # logs
            msg = 'it: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}' \
                  ''.format(i, v_table[idx_state_init], v_rms_errors[j])
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

    # initialize environments
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # set deterministic policy from the hjb control
    policy = np.array([
        env.get_action_idx(sol_hjb.u_opt[idx_state])
        for idx_state, _ in enumerate(env.state_space_h)
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
