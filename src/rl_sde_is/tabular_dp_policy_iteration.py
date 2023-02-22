import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.dynammic_programming import compute_p_tensor_batch, compute_r_table
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import plot_value_function_1d, plot_det_policy_1d
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def policy_iteration(env, gamma=1.0, n_iterations=100, n_avg_iterations=10, load=False):
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
        data = load_data(dir_path)
        return data

    # compute p tensor and r table
    p_tensor = compute_p_tensor_batch(env)
    r_table = compute_r_table(env)

    # initialize value function table and policy (array storing the indices of the actions)
    v_table = np.zeros(env.n_states)
    policy = np.random.randint(env.n_actions, size=env.n_states)

    # set values for the target set
    v_table[env.idx_ts] = 0
    policy[env.idx_ts] = env.idx_null_action

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init).item()

    # in each value iteration and policy update
    for i in np.arange(n_iterations):

        # value iteration

        # copy value function table
        v_table_i = v_table.copy()

        # loop over states not in the target set
        for idx_state in range(env.idx_lb):

            idx_action = policy[idx_state]

            # Bellman expectation equation
            v_table[idx_state] = r_table[idx_state, idx_action] \
                               + gamma * np.dot(
                                   p_tensor[np.arange(env.n_states), idx_state, idx_action],
                                   v_table_i[np.arange(env.n_states)],
                               )

        # policy update

        # loop over states not in the target set
        for idx_state in range(env.idx_lb):

            # preallocate values
            values = np.zeros(env.n_actions)

            # loop over all actions
            for idx_action in range(env.n_actions):

                values[idx_action] = r_table[idx_state, idx_action] \
                                   + gamma \
                                   * np.dot(
                                       p_tensor[np.arange(env.n_states), idx_state, idx_action],
                                       v_table_i[np.arange(env.n_states)],
                                   )

            # Bellman optimality equation
            policy[idx_state] = np.argmax(values)



        # logs
        if i % n_avg_iterations == 0:
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, v_table[idx_state_init])
            print(msg)

    # compute policy actions
    policy = env.action_space_h[policy]

    data = {
        'n_iterations': n_iterations,
        'v_table' : v_table,
        'policy' : policy,
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

    # run dynamic programming value iteration
    data = policy_iteration(
        env,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        n_avg_iterations=args.n_avg_iterations,
        load=args.load,
    )

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # do plots
    plot_value_function_1d(env, data['v_table'], sol_hjb.value_function)
    plot_det_policy_1d(env, data['policy'], sol_hjb.u_opt)


if __name__ == '__main__':
    main()
