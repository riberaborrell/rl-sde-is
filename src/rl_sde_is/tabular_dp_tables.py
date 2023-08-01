import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.dynamic_programming import compute_p_tensor_batch, compute_r_table
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.tabular_methods import compute_tables
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def check_p_tensor(env, p_tensor):
    sum_probs = np.sum(p_tensor, axis=0)
    return np.isclose(sum_probs, 1).all()

def compute_optimal_v_table(env, r_table, p_tensor, value_function_opt, policy_opt):

    # get states and actions
    states_idx = np.arange(env.n_states)
    actions_idx = env.get_action_idx(np.expand_dims(policy_opt, axis=1))

    # bellman expectation equation following optimal policy
    d = np.where(env.is_in_ts, 1, 0)
    v_table = (1 - d) * np.matmul(
       p_tensor[:, states_idx, actions_idx].T,
       value_function_opt,
    ) + r_table[states_idx, actions_idx]

    return v_table

def compute_optimal_q_table(env, r_table, p_tensor, value_function_opt, policy_opt):

    # bellman expectation equation following optimal policy
    d = np.expand_dims(np.where(env.is_in_ts, 1, 0), axis=1)
    q_table = (1 - d) * np.dot(
        np.moveaxis(p_tensor, 0, -1),
        value_function_opt,
    ) + r_table
    return q_table

def dynamic_programming_tables(env, value_function_opt=None, policy_opt=None, load=False):

    # get dir path
    rel_dir_path = get_dynamic_programming_tables_dir_path(env)

    # load results
    if load:
        return load_data(rel_dir_path)

    # compute r table and p tensor
    r_table = compute_r_table(env)
    p_tensor = compute_p_tensor_batch(env)

    # check
    assert check_p_tensor(env, p_tensor)

    v_table = compute_optimal_v_table(env, r_table, p_tensor, value_function_opt, policy_opt)
    q_table = compute_optimal_q_table(env, r_table, p_tensor, value_function_opt, policy_opt)

    #array = np.isclose(v_table, np.max(q_table, axis=1))

    data = {
        'r_table': r_table,
        'p_tensor': p_tensor,
        'q_table': q_table,
        'rel_dir_path': rel_dir_path,
    }
    save_data(data, rel_dir_path)
    return data


def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # discretize observation and action space
    env.set_action_space_bounds()
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    value_function_opt = -sol_hjb.value_function
    policy_opt = sol_hjb.u_opt

    # compute dp tables
    data = dynamic_programming_tables(env, value_function_opt, policy_opt, load=args.load)

    # plot
    if not args.plot:
        return

    r_table = data['r_table']
    p_tensor = data['p_tensor']
    q_table = data['q_table']
    v_table, a_table, policy = compute_tables(env, q_table)

    # plot reward table and p_tensor
    plot_reward_table(env, r_table)

    #for i in np.arange(env.n_states):
    #    plot_reward_table(env, p_tensor[:, i, :])

    # plot value function and reward following the optimal policy
    plot_value_function_1d(env, v_table, value_function_opt)
    states_idx = np.arange(env.n_states)
    actions_idx = env.get_action_idx(np.expand_dims(policy_opt, axis=1))
    plot_reward_following_policy(env, r_table[states_idx, actions_idx])

    # plot q value function and advantage value
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table, policy_opt=policy_opt, policy_critic=policy)

if __name__ == '__main__':
    main()
