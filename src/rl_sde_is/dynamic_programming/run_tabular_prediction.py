import numpy as np
import gymnasium as gym

import gym_sde_is
from gym_sde_is.wrappers.tabular_env import TabularEnv

from rl_sde_is.utils.tabular_methods import compute_rms_error
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.path import get_dynamic_programming_tables_dir_path, \
                                 get_dynamic_programming_dir_path, save_data, load_data
from rl_sde_is.utils.plots import *

def value_table_update(env, r_table, p_tensor, policy, v_table, gamma):

    # copy value function table
    v_table_i = v_table.copy()
    d = np.where(env.is_target_set(env.state_space_h), 1, 0).squeeze()

    # loop over state indices
    for state_idx in range(env.n_states):

        # choose action following policy
        action_idx = policy[state_idx]

        # update v table
        v_table[state_idx] = r_table[state_idx, action_idx]

        # loop over next state indices
        for next_state_idx in range(env.n_states):

            # update v table
            v_table[state_idx] += (1 - d[state_idx]) * gamma \
                               * p_tensor[next_state_idx, state_idx, action_idx] \
                               * v_table_i[next_state_idx]

def value_table_update_semi_vect(env, r_table, p_tensor, policy, v_table, gamma):
    v_table_i = v_table.copy()
    d = np.where(env.is_target_set(env.state_space_h), 1, 0).squeeze()
    for state_idx in range(env.n_states):
        action_idx = policy[state_idx]
        v_table[state_idx] = r_table[state_idx, action_idx] \
                           + gamma * (1 - d[state_idx]) * np.dot(
                                p_tensor[:, state_idx, action_idx].squeeze(),
                                v_table_i,
                           )

def value_table_update_vect(env, r_table, p_tensor, policy, v_table, gamma):
    actions_idx = policy.squeeze()
    states = env.state_space_h.flatten()
    d = np.where(env.is_target_set(states), 1, 0)
    v_table = r_table[np.arange(env.n_states), actions_idx] \
            + gamma * (1 - d) * np.matmul(
                p_tensor[:, np.arange(env.n_states), actions_idx].T,
                v_table
            )
    return v_table

def policy_evaluation(env, gamma=1.0, n_iterations=100, eval_freq=10,
                      policy=None, value_function_opt=None, live_plot_freq=None, load=False):

    ''' Dynamic programming policy evaluation.
    '''
    # get dir path
    dir_path = get_dynamic_programming_dir_path(
        env,
        agent='dp-prediction',
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
    v_table = - np.random.rand(env.n_states)

    # preallocate value function rms errors
    n_test_iterations = n_iterations // eval_freq + 1
    v_rms_errors = np.empty(n_test_iterations)

    # compute errors
    v_rms_errors[0] = compute_rms_error(value_function_opt, v_table)

    # initialize live figures
    if live_plot_freq:
        line = initialize_value_function_1d_figure(env, v_table, value_function_opt)

    # for each iteration
    for i in np.arange(n_iterations):

        #value_table_update(env, r_table, p_tensor, policy, v_table, gamma)
        #value_table_update_semi_vect(env, r_table, p_tensor, policy, v_table, gamma)
        v_table = value_table_update_vect(env, r_table, p_tensor, policy, v_table, gamma)

        # test
        if (i + 1) % eval_freq == 0:

            # compute root mean square error of value function
            j = (i + 1) // eval_freq
            v_rms_errors[j] = compute_rms_error(value_function_opt, v_table)

            # logs
            msg = 'it: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}' \
                  ''.format(i+1, v_table[env.state_init_idx].item(), v_rms_errors[j])
            print(msg)

            # update live figure
            if live_plot_freq and (i + 1) % live_plot_freq == 0:
                update_value_function_1d_figure(env, v_table, line)

    data = {
        'n_iterations': n_iterations,
        'v_table' : v_table,
        'v_rms_errors' : v_rms_errors,
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

    # set deterministic policy from the hjb control
    policy_indices = env.get_action_idx(sol_hjb.u_opt)[0]

    # run dynammic programming policy evaluation of the optimal policy
    data = policy_evaluation(
        env,
        policy=policy_indices,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        eval_freq=args.eval_freq,
        value_function_opt=-sol_hjb.value_function,
        live_plot_freq=args.live_plot_freq,
        load=args.load,
    )

    # plot
    if not args.plot:
        return

    # do plots
    policy = env.action_space_h[policy_indices]
    plot_det_policy_1d(env, policy, sol_hjb.u_opt)
    plot_value_function_1d(env, data['v_table'], -sol_hjb.value_function)
    plot_value_rms_error_iterations(data['v_rms_errors'], args.eval_freq)

if __name__ == '__main__':
    main()
