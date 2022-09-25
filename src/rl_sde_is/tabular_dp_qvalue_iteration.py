import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.dynammic_programming import compute_p_tensor_batch, compute_r_table
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import compute_tables
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def qvalue_iteration(env, gamma=1.0, n_iterations=100, n_avg_iterations=10,
                     value_function_hjb=None, control_hjb=None, load=False, plot=False):
    ''' Dynamic programming q-value iteration.
    '''

    # get dir path
    rel_dir_path = get_dynamic_programming_dir_path(
        env,
        agent='dp-q-value-iteration',
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
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set and null action
    q_table[env.idx_lb:env.idx_rb+1] = 0

    # get index x_init
    idx_state_init = env.get_state_idx(env.state_init)

    # initialize live figures
    if plot:
        v_table, a_table, policy = compute_tables(env, q_table)
        lines = initialize_q_learning_figures(env, q_table, v_table, a_table, policy,
                                          value_function_hjb, control_hjb)

    # for each iteration
    for i in np.arange(n_iterations):

        # copy value function table
        q_table_i = q_table.copy()

        # loop over states not in the target set 
        for idx_state in range(env.n_states):

            # loop over all actions
            for idx_action in range(env.n_actions):
                value = r_table[idx_state, idx_action] \
                      + gamma \
                      * np.dot(
                          p_tensor[np.arange(env.n_states), idx_state, idx_action],
                          np.max(q_table_i, axis=1),
                        )

                q_table[idx_state, idx_action] = value

        # logs
        if i % n_avg_iterations == 0:
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, np.max(q_table[idx_state_init]))
            print(msg)

        # update live figures
        if i % 10 == 0:
            v_table, a_table, policy = compute_tables(env, q_table)
            update_q_learning_figures(env, q_table, v_table, a_table, policy, lines)


    data = {
        'n_iterations': n_iterations,
        'q_table' : q_table,
    }
    save_data(data, rel_dir_path)
    return data


def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D(dt=args.dt)

    # discretize state and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run dynamic programming tabular method 
    data = qvalue_iteration(
        env,
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        n_avg_iterations=args.n_avg_iterations,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )

    # plot
    if not args.plot:
        return

    # compute tables
    q_table = data['q_table']
    v_table, a_table, policy_greedy = compute_tables(env, q_table)

    # do plots
    plot_q_value_function(env, q_table)
    plot_value_function(env, v_table, sol_hjb.value_function)
    plot_advantage_function(env, a_table)
    plot_det_policy(env, policy_greedy, sol_hjb.u_opt)


if __name__ == '__main__':
    main()
