import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.tabular_methods import *
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def test_q(test_env, q_table, batch_size=100):
    returns = np.zeros(batch_size)
    time_steps = np.zeros(batch_size)

    # reset environment
    states = test_env.reset(batch_size)

    # terminal state flag
    done = np.full((batch_size, 1), False)

    # not arrived in target set yet
    idx = np.where(done == False)[0]

    # number of trajectories not in the target set
    n_not_in_ts = idx.shape[0]

    while n_not_in_ts > 0:

        # get index of the state
        idx_states = test_env.get_states_idx_vectorized(states)

        idx_actions, actions = get_epsilon_greedy_actions_vectorized(test_env, q_table, idx_states, 0.)

        # step dynamics forward
        states, rewards, done = test_env.step_vectorized_stopped(states, actions, idx)

        # update returns and time steps
        returns += np.squeeze(rewards)
        time_steps += 1

        # not arrived in target set yet
        idx = np.where(done == False)[0]

        # number of trajectories not in the target set
        n_not_in_ts = idx.shape[0]

    return np.mean(returns), np.mean(time_steps)

def q_learning(env, gamma=1., batch_size=10, lr=0.01,
               n_epochs=100, n_avg_epochs=1, n_steps_per_epoch=5000,
               eps_type='linear-decay', eps_init=1., eps_min=0., eps_decay=0.98,
               value_function_hjb=None, control_hjb=None, load=False):

    # get dir path
    dir_path = get_qlearning_batch_dir_path(
        env,
        agent='q-learning-batch',
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        eps_type=eps_type,
        eps_init=eps_init,
        eps_min=eps_min,
    )

    # load results
    if load:
        data = load_data(dir_path)
        return data

    # initialize frequency and q-value function table
    n_table = np.zeros((env.n_states, env.n_actions), dtype=np.int32)
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set
    q_table[env.idx_lb:env.idx_rb+1] = 0

    # set epsilons
    epsilons = set_epsilons(
        eps_type,
        n_episodes=n_epochs,
        eps_init=eps_init,
        eps_min=eps_min,
        eps_decay=eps_decay,
    )
    # initialize plots 
    #images, lines = initialize_figures(env, n_table, q_table, n_episodes,
    #                                   value_function_hjb, control_hjb)

    # preallocate returns and time steps
    test_returns = np.empty(n_epochs)
    test_time_steps = np.empty(n_epochs)

    # fill with nan values (for plotting purposes only)
    test_returns.fill(np.nan)
    test_time_steps.fill(np.nan)

    # initialize test environment
    test_env = DoubleWellStoppingTime1D()

    # discretize observation and action space
    test_env.discretize_state_space(env.h_state)
    test_env.discretize_action_space(env.h_action)

    # for each episode
    for ep in np.arange(n_epochs):

        # reset environment
        states = env.reset_vectorized(batch_size)

        # terminal state flag
        done = np.full((batch_size, 1), False)

        # sample episode
        for k in np.arange(n_steps_per_epoch):

            # reset if state is in a terminal state
            states = np.where(done, env.reset_vectorized(batch_size), states)

            # get index of the state
            idx_states = env.get_states_idx_vectorized(states)

            # get epsilon
            epsilon = epsilons[ep]

            # choose action following epsilon greedy action
            idx_actions, actions = get_epsilon_greedy_actions_vectorized(env, q_table, idx_states, epsilon)

            # step dynamics forward
            new_states, rewards, done = env.step_vectorized(states, actions)
            idx_new_states = env.get_states_idx_vectorized(new_states)

            # update q values
            n_table[idx_states, idx_actions] += 1
            q_table[idx_states, idx_actions] += lr * (
                  np.squeeze(rewards) \
                + gamma * np.max(q_table[idx_new_states], axis=1) \
                - q_table[idx_states, idx_actions]
            )

            # update state
            states = new_states

        # test greedy policy following the actual q_table
        #test_returns[ep], test_time_steps[ep] = test_q(test_env, q_table)

        # update plots
        #update_figures(env, n_table, q_table, returns, avg_returns, time_steps,
        #               avg_time_steps, images, lines)

        # logs
        if ep % n_avg_epochs == 0:
            msg = 'ep: {:3d}, V(s_init): {:.3f}, test return {:2.2f}, ' \
                  'test time steps: {:2.2f}, epsilon: {:2.3f}'.format(
            #msg = 'ep: {:3d}, V(s_init): {:.3f}, ' \
            #      'epsilon: {:2.3f}'.format(
                    ep,
                    np.max(q_table[env.idx_state_init]),
                    test_returns[ep],
                    test_time_steps[ep],
                    epsilon,
                )
            print(msg)

    data = {
        'n_epochs': n_epochs,
        'test_returns': test_returns,
        'test_time_steps': test_time_steps,
        'n_table' : n_table,
        'q_table' : q_table,
    }
    save_data(dir_path, data)
    return data


def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run mc learning algorithm
    data = q_learning(
        env,
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.n_epochs,
        n_steps_per_epoch=args.n_steps_per_epoch,
        eps_type=args.eps_type,
        eps_init=args.eps_init,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
    )
    n_table, q_table = data['n_table'], data['q_table']

    # compute tables
    v_table, a_table, policy = compute_tables(env, q_table)

    # do plots
    plot_frequency(env, n_table)
    plot_q_value_function(env, q_table)
    plot_value_function(env, v_table, sol_hjb.value_function)
    plot_advantage_function(env, a_table)
    plot_det_policy(env, policy, sol_hjb.u_opt)


if __name__ == '__main__':
    main()
