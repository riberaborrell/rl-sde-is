import numpy as np
import gymnasium as gym

import gym_sde_is
from gym_sde_is.wrappers.tabular_env import TabularEnv

from rl_sde_is.dynamic_programming.dp_utils import compute_p_tensor_batch, compute_r_table
from rl_sde_is.utils.tabular_methods import compute_value_advantage_and_greedy_policy
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.path import get_dynamic_programming_tables_dir_path, save_data, load_data
from rl_sde_is.utils.plots import *

def check_p_tensor(env, p_tensor):
    sum_probs = np.sum(p_tensor, axis=0)
    return np.isclose(sum_probs, 1).all()

def compute_optimal_v_table(env, r_table, p_tensor, value_function_opt, policy_opt):

    # get states and actions
    states_idx = np.arange(env.n_states)
    actions_idx = env.get_action_idx(policy_opt)

    # bellman expectation equation following optimal policy
    d = np.where(env.is_target_set(env.state_space_h.flatten()), 1, 0)[0]
    v_table = (1 - d) * np.matmul(
       p_tensor[:, states_idx, actions_idx].T,
       value_function_opt,
    ) + r_table[states_idx, actions_idx]

    return v_table

def compute_optimal_q_table(env, r_table, p_tensor, value_function_opt):

    # bellman expectation equation following optimal policy
    d = np.where(env.is_target_set(env.state_space_h), 1, 0)
    q_table = (1 - d) * np.dot(
        np.moveaxis(p_tensor, 0, -1),
        value_function_opt,
    ) + r_table
    return q_table

def dynamic_programming_tables(env, value_function_opt=None, policy_opt=None, load=False):

    assert env.d == 1, 'only 1-dimensional problems is supported'

    # get dir path
    dir_path = get_dynamic_programming_tables_dir_path(env)

    # load results
    if load:
        return load_data(dir_path)

    # compute r table and p tensor
    r_table = compute_r_table(env)
    p_tensor = compute_p_tensor_batch(env)

    # check
    assert check_p_tensor(env, p_tensor)

    v_table = compute_optimal_v_table(env, r_table, p_tensor, value_function_opt, policy_opt)
    q_table = compute_optimal_q_table(env, r_table, p_tensor, value_function_opt)

    #array = np.isclose(v_table, np.max(q_table, axis=1))

    data = {
        'dt': env.dt,
        'h_state': env.h_state,
        'h_action': env.h_action,
        'r_table': r_table,
        'p_tensor': p_tensor,
        'v_table': v_table,
        'q_table': q_table,
        'dir_path': dir_path,
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
    value_function_opt = - sol_hjb.value_function
    policy_opt = sol_hjb.u_opt

    # compute dp tables
    data = dynamic_programming_tables(env, value_function_opt, policy_opt, load=args.load)

    # plot
    if not args.plot:
        return

    r_table = data['r_table']
    p_tensor = data['p_tensor']
    q_table = data['q_table']
    v_table, a_table, policy = compute_value_advantage_and_greedy_policy(env, q_table)

    # plot reward table and p_tensor
    plot_reward_table(env, r_table)

    #for i in np.arange(env.n_states):
    #    plot_reward_table(env, p_tensor[:, i, :])

    # plot value function and reward following the optimal policy
    plot_value_function_1d(env, v_table, value_function_opt)
    states_idx = np.arange(env.n_states)
    actions_idx = env.get_action_idx(policy_opt)[0]
    plot_reward_following_policy(env, r_table[states_idx, actions_idx])

    # plot q value function and advantage value
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table, policy_opt=policy_opt, policy_critic=policy)

if __name__ == '__main__':
    main()
