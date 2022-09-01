import numpy as np

from base_parser import get_base_parser
from environments import DoubleWellStoppingTime1D
from tabular_learning import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def test_q(test_env, q_table, batch_size=100):
    returns = np.zeros(batch_size)
    time_steps = np.zeros(batch_size)

    # reset environment
    states = test_env.reset_vectorized(batch_size)

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

def q_learning(env, gamma=1., batch_size=10, epsilons=None, alpha=0.01,
               n_epochs=100, n_avg_epochs=1, n_steps_per_epoch=5000,
               value_function_hjb=None, control_hjb=None):

    # initialize frequency and q-value function table
    n_table = np.zeros((env.n_states, env.n_actions), dtype=np.int32)
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set
    q_table[env.idx_lb:env.idx_rb+1] = 0

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
            q_table[idx_states, idx_actions] += alpha * (
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

    return n_table, q_table


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

    # set epsilons
    #epsilons = get_epsilons_constant(args.n_episodes, eps_init=0.1)
    #epsilons = get_epsilons_harmonic(args.n_episodes)
    epsilons = get_epsilons_linear_decay(args.n_epochs, args.eps_min, exploration=0.5)
    #epsilons = get_epsilons_exp_decay(args.n_episodes, args.eps_init, args.eps_decay)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run mc learning algorithm
    info = q_learning(
        env,
        gamma=args.gamma,
        batch_size=args.batch_size,
        epsilons=epsilons,
        alpha=args.alpha,
        n_epochs=args.n_epochs,
        n_steps_per_epoch=args.n_steps_per_epoch,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
    )

    n_table, q_table = info

    # do plots
    plot_frequency_table(env, n_table)
    plot_q_table(env, q_table)
    plot_v_table(env, q_table, sol_hjb.value_function)
    plot_a_table(env, q_table)
    plot_greedy_policy(env, q_table, sol_hjb.u_opt)


if __name__ == '__main__':
    main()
