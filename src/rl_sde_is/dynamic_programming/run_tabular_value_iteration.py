import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.wrappers.tabular_env import TabularEnv

from rl_sde_is.utils.tabular_methods import compute_rms_error
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.path import get_dynamic_programming_tables_dir_path, \
                                 get_dynamic_programming_dir_path, save_data, load_data
from rl_sde_is.utils.plots import *

def v_table_update_semi_vect(env, r_table, p_tensor, v_table, gamma):

    # copy value function table
    v_table_i = v_table.copy()

    d = np.where(env.is_target_set(env.state_space_h), 1, 0).squeeze()

    # loop over states not in the target set
    for state_idx in range(env.n_states):

        # preallocate values
        values = np.zeros(env.n_actions)

        # loop over all actions
        for action_idx in range(env.n_actions):

            values[action_idx] = r_table[state_idx, action_idx] \
                               + gamma * (1 - d[state_idx]) \
                               * np.dot(
                                   p_tensor[np.arange(env.n_states), state_idx, action_idx],
                                   v_table_i[np.arange(env.n_states)],
                                )

        # Bellman optimality equation
        v_table[state_idx] = np.max(values)

def v_table_update_vect(env, r_table, p_tensor, v_table, gamma):

    d = np.where(env.is_target_set(env.state_space_h), 1, 0)

    values = r_table \
           + gamma * (1 - d) * np.matmul(
                np.swapaxes(p_tensor, 0, 2),
                v_table,
            ).transpose()

    # Bellman optimality equation
    return np.max(values, axis=1)


def value_iteration(env, gamma=1.0, n_iterations=100, eval_freq=10,
                    value_function_opt=None, policy_opt=None, live_plot_freq=None, load=False):
    from rl_sde_is.dynamic_programming.run_tabular_policy_iteration import policy_update_semi_vect, \
                                                                           policy_update_vect

    ''' Dynamic programming value iteration with synchronous back ups.
    '''
    # get dir path
    dir_path = get_dynamic_programming_dir_path(
        env,
        agent='dp-value-iterations',
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

    if live_plot_freq:
        v_line = initialize_value_function_1d_figure(env, v_table, value_function_opt)

    # value iteration
    for i in np.arange(n_iterations):

        #v_table_update_semi_vect(env, r_table, p_tensor, v_table, gamma)
        v_table = v_table_update_vect(env, r_table, p_tensor, v_table, gamma)

        # test
        if (i + 1) % eval_freq == 0:

            # compute root mean square error of value function and policy
            j = (i + 1) // eval_freq
            v_rms_errors[j] = compute_rms_error(value_function_opt, v_table)

            # logs
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, v_table[env.state_init_idx].item())
            print(msg)

            # update live figures
            if live_plot_freq and (i + 1) % live_plot_freq == 0:
                update_value_function_1d_figure(env, v_table, v_line)

    # policy computation.
    #policy_indices = policy_update_semi_vect(env, r_table, p_tensor, v_table, gamma)
    policy_indices = policy_update_vect(env, r_table, p_tensor, v_table, gamma)

    # compute policy actions
    policy = env.action_space_h[policy_indices]

    data = {
        'n_iterations': n_iterations,
        'v_table' : v_table,
        'policy' : policy,
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

    # run dynamic programming value iteration
    data = value_iteration(
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

    # do plots
    plot_value_function_1d(env, data['v_table'], -sol_hjb.value_function)
    plot_det_policy_1d(env, data['policy'], sol_hjb.u_opt)
    plot_value_rms_error_iterations(data['v_rms_errors'], args.eval_freq)

if __name__ == '__main__':
    main()
