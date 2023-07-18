import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.dynammic_programming import compute_p_tensor_batch, compute_r_table
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def check_p_tensor(env, p_tensor):
    sum_probs = np.sum(p_tensor, axis=0)
    test = np.isclose(sum_probs, 1).all()
    print(test)

def check_bellman_equation(env, r_table, p_tensor, value_function_opt, policy_opt):

    # get states and actions
    states_idx = np.arange(env.n_states)
    actions_idx = env.get_action_idx(np.expand_dims(policy_opt, axis=1))

    d = np.where(env.is_in_ts, 1, 0)

    # bellman expectation equation following optimal policy
    value_function_bell = (1 - d) * np.matmul(
       p_tensor[:, states_idx, actions_idx].T,
       value_function_opt,
    ) + r_table[states_idx, actions_idx]

    plot_value_function_1d(env, value_function_bell, value_function_opt)
    plot_reward_following_policy(env, r_table[states_idx, actions_idx])


def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # discretize observation and action space
    env.set_action_space_bounds()
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get dir path
    rel_dir_path = get_dynamic_programming_tables_dir_path(env)

    # load results
    if args.load:
        data = load_data(rel_dir_path)

    # compute r table and p tensor
    else:
        data = {
            'r_table': compute_r_table(env),
            'p_tensor': compute_p_tensor_batch(env),
            'rel_dir_path': rel_dir_path,
        }
        save_data(data, rel_dir_path)

    r_table = data['r_table']
    p_tensor = data['p_tensor']

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # checks
    check_p_tensor(env, p_tensor)
    check_bellman_equation(env, r_table, p_tensor, -sol_hjb.value_function, sol_hjb.u_opt)

    # plot
    if not args.plot:
        return

    # plot reward table and p_tensor
    plot_reward_table(env, r_table)
    for i in np.arange(env.n_states):
        plot_reward_table(env, p_tensor[:, i, :])



if __name__ == '__main__':
    main()
