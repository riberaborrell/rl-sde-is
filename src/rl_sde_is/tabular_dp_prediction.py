import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.dynammic_programming import compute_p_tensor_batch, compute_r_table
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import plot_value_function
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def policy_evaluation(env, policy, gamma=1.0, n_iterations=100, n_avg_iterations=10, load=False):
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
        data = load_data(dir_path)
        return data

    # compute p tensor and r table
    p_tensor = compute_p_tensor_batch(env)
    r_table = compute_r_table(env)

    # initialize value function table
    v_table = - np.random.rand(env.n_states)

    # set values for the target set
    v_table[env.idx_lb:env.idx_rb+1] = 0

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init)

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

        # logs
        if i % n_avg_iterations == 0:
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, v_table[idx_state_init])
            print(msg)

    data = {
        'n_iterations': n_iterations,
        'v_table' : v_table,
    }
    save_data(dir_path, data)

    return data

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D(dt=args.dt)

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

    # run mc learning agent following optimal policy
    data = policy_evaluation(
        env,
        policy,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        n_avg_iterations=args.n_avg_iterations,
        load=args.load,
    )

    # do plots
    plot_value_function(env, data['v_table'], sol_hjb.value_function)


if __name__ == '__main__':
    main()
