import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.wrappers.tabular_env import TabularEnv

from rl_sde_is.utils.tabular_methods import compute_value_advantage_and_greedy_policy, compute_rms_error
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.path import get_dynamic_programming_tables_dir_path, \
                                 get_dynamic_programming_dir_path, save_data, load_data
from rl_sde_is.utils.plots import *


def q_table_update_semi_vect(env, r_table, p_tensor, q_table, gamma):

    # copy value function table
    q_table_i = q_table.copy()

    d = np.where(env.is_target_set(env.state_space_h), 1, 0).squeeze()

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

    return q_table

def q_table_update_vect(env, r_table, p_tensor, q_table, gamma):

    d = np.where(env.is_target_set(env.state_space_h), 1, 0)
    q_table = r_table \
            + (1 - d) * gamma * np.matmul(
                np.swapaxes(p_tensor, 0, 2),
                np.max(q_table, axis=1)
            ).transpose()

    return q_table

def qvalue_iteration(env, gamma=1.0, n_iterations=100, eval_freq=10,
                     value_function_opt=None, policy_opt=None, live_plot_freq=False, load=False):

    ''' Dynamic programming q-value iteration.
    '''
    # get dir path
    dir_path = get_dynamic_programming_dir_path(
        env,
        agent='dp-q-value-iteration',
        n_iterations=n_iterations,
    )

    # load results
    if load:
        return load_data(dir_path)

    # load dp tables
    tables_data = load_data(get_dynamic_programming_tables_dir_path(env))
    r_table = tables_data['r_table']
    p_tensor = tables_data['p_tensor']

    # initialize value function table
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # preallocate value function rms errors
    n_test_iterations = n_iterations // eval_freq + 1
    v_rms_errors = np.empty(n_test_iterations)
    p_rms_errors = np.empty(n_test_iterations)

    # compute tables
    v_table, a_table, policy = compute_value_advantage_and_greedy_policy(env, q_table)

    # compute errors
    v_rms_errors[0] = compute_rms_error(value_function_opt, v_table)
    p_rms_errors[0] = compute_rms_error(policy_opt, policy)

    # initialize live figures
    if live_plot_freq:
        lines = initialize_tabular_figures(env, q_table, v_table, a_table, policy,
                                           value_function_opt, policy_opt)
    # for each iteration
    for i in np.arange(n_iterations):

        #q_table_update_semi_vect(env, r_table, p_tensor, q_table, gamma)
        q_table = q_table_update_vect(env, r_table, p_tensor, q_table, gamma)

        # test
        if (i + 1) % eval_freq == 0:

            # compute tables
            v_table, a_table, policy = compute_value_advantage_and_greedy_policy(env, q_table)

            # compute root mean square error of value function and policy
            j = (i + 1) // eval_freq
            v_rms_errors[j] = compute_rms_error(value_function_opt, v_table)
            p_rms_errors[j] = compute_rms_error(policy_opt, policy)

            # logs
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i+1, np.max(q_table[env.state_init_idx]))
            print(msg)

            # update live figures
            if live_plot_freq and (i + 1) % live_plot_freq == 0:
                update_tabular_figures(env, q_table, v_table, a_table, policy, lines)

    data = {
        'n_iterations': n_iterations,
        'q_table' : q_table,
        'v_rms_errors' : v_rms_errors,
        'p_rms_errors' : p_rms_errors,
    }
    save_data(data, dir_path)
    return data


def main():
    args = get_base_parser().parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        state_init_dist=args.state_init_dist,
        reward_type=args.reward_type,
        baseline_scale_factor=args.baseline_scale_factor,
    )
    env = TabularEnv(env, args.h_state, args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    sol_hjb.coarse_solution(args.h_state)

    # run dynamic programming q-value iteration
    data = qvalue_iteration(
        env,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        eval_freq=args.eval_freq,
        value_function_opt=-sol_hjb.value_function,
        policy_opt=sol_hjb.u_opt,
        live_plot_freq=args.live_plot_freq,
        load=args.load,
    )

    # plot
    if not args.plot:
        return

    # compute tables
    q_table = data['q_table']
    v_table, a_table, policy_greedy = compute_value_advantage_and_greedy_policy(env, q_table)

    # do plots
    plot_value_function_1d(env, v_table, -sol_hjb.value_function)
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table)
    plot_det_policy_1d(env, policy_greedy, sol_hjb.u_opt)
    plot_value_rms_error_iterations(data['v_rms_errors'], args.eval_freq)
    plot_policy_rms_error_iterations(data['p_rms_errors'], args.eval_freq)

if __name__ == '__main__':
    main()
