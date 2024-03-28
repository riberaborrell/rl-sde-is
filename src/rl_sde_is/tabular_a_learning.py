import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import *
from rl_sde_is.utils_numeric import discount_cumsum
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def a_learning(env, gamma=1., epsilons=None, lr=0.01, n_episodes=1000,
               n_steps_lim=1000, test_freq_episodes=10, n_avg_episodes=10, seed=None,
               value_function_opt=None, policy_opt=None, load=False, live_plot=False):
    ''' Advantage learning

        A(s, a) =max_{a'}(Q(s, a')) + (Q(s, a) - max_{a'}(Q(s, a))) * k / Delta t
        What is k?
    '''
    alpha = 1

    # get dir path
    rel_dir_path = get_mc_learning_dir_path(
        env,
        agent='a-learning',
        eps_type='constant',
        eps_init=0,
        n_episodes=n_episodes,
        lr=lr,
        lam=1,
        seed=seed,
    )

    # load results
    if load:
        data = load_data(rel_dir_path)
        return data

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # initialize frequency and q-value function table
    n_table = np.zeros((env.n_states, env.n_actions), dtype=np.int32)
    v_table = - np.random.rand(env.n_states)
    a_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set
    v_table[env.ts_idx] = 0
    a_table[env.ts_idx] = 0

    # preallocate returns and time steps
    returns = np.empty(n_episodes)
    avg_returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    avg_time_steps = np.empty(n_episodes)

    # preallocate value function and control rms errors
    n_test_episodes = n_episodes // test_freq_episodes + 1
    v_rms_errors = np.empty(n_test_episodes)
    p_rms_errors = np.empty(n_test_episodes)


    # initialize live figures
    if live_plot:
        actions_idx = np.argmax(a_table, axis=1)
        greedy_actions = env.action_space_h[actions_idx]
        im, line = initialize_advantage_function_1d_figure(env, a_table, policy_opt, greedy_actions)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # reset trajectory rewards
        rewards = np.empty(0)

        # terminal state flag
        done = False

        # get epsilon
        epsilon = epsilons[ep]

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # get index of the state
            idx_state = env.get_state_idx(state)

            # choose action following epsilon greedy action
            idx_action, action = get_epsilon_greedy_action(env, a_table, idx_state, epsilon)

            # get idx state-action pair
            idx = (idx_state, idx_action,)

            # step dynamics forward
            new_state, r, done, _ = env.step(state, action)
            idx_new_state = env.get_state_idx(new_state)

            # update frequency table
            n_table[idx] += 1

            # advantage error
            max_a_table = np.max(a_table[idx_state])
            d = np.where(done, 1., 0.)
            a_error = max_a_table \
                    + (r + gamma * (1 - d) * v_table[idx_new_state] - v_table[idx_state]) / env.dt \
                    - a_table[idx]

            # update a table
            a_table[idx] += lr * a_error

            # value error
            max_a_table_new = np.max(a_table[idx_state])
            #v_error = v_table[idx_state] + (max_a_table_new - max_a_table) / alpha
            v_error = (max_a_table_new - max_a_table) / alpha

            # update v table
            v_table[idx_state] = v_table[idx_state] + lr * v_error

            # normalize step

            # save action and reward
            rewards = np.append(rewards, r)

            # update state
            state = new_state

        # compute return
        ep_returns = discount_cumsum(rewards, gamma)

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            idx_last_episodes = slice(0, ep + 1)
        else:
            idx_last_episodes = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        returns[ep] = ep_returns[0]
        avg_returns[ep] = np.mean(returns[idx_last_episodes])
        time_steps[ep] = k
        avg_time_steps[ep] = np.mean(time_steps[idx_last_episodes])

        # test
        if (ep + 1) % test_freq_episodes == 0:

            # compute root mean square error of value function and control
            ep_test = (ep + 1) // test_freq_episodes
            #v_table, a_table, policy = compute_tables(env, q_table)
            #v_rms_errors[ep_test] = compute_rms_error(value_function_opt, v_table)
            #p_rms_errors[ep_test] = compute_rms_error(policy_opt, policy)

            # logs
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                    'run avg time steps: {:2.2f}, epsilon: {:.2f}'.format(
                    ep,
                    #np.max(q_table[idx_state_init]),
                    np.nan,
                    avg_returns[ep],
                    avg_time_steps[ep],
                    epsilon,
                )
            print(msg)

            # update live figures
            if live_plot:
                actions_idx = np.argmax(a_table, axis=1)
                greedy_actions = env.action_space_h[actions_idx]
                update_advantage_function_1d_figure(env, a_table, greedy_actions, im, line)

    data = {
        'gamma': gamma,
        'n_episodes': n_episodes,
        'n_steps_lim': n_steps_lim,
        'lr': lr,
        'seed': seed,
        'n_avg_episodes' : n_avg_episodes,
        'test_freq_episodes' : test_freq_episodes,
        'returns': returns,
        'avg_returns': avg_returns,
        'time_steps': time_steps,
        'avg_time_steps': avg_time_steps,
        'n_table' : n_table,
        'a_table' : a_table,
        'v_rms_errors' : v_rms_errors,
        'p_rms_errors' : p_rms_errors,
    }
    save_data(data, rel_dir_path)

    return data


def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # set epsilons
    #epsilons = get_epsilons_constant(args.n_episodes, eps_init=0.)
    epsilons = get_epsilons_constant(args.n_episodes, eps_init=1.)
    #epsilons = get_epsilons_linear_decay(args.n_episodes, eps_min=0.01, exploration=0.5)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run advantage learning algorithm
    data = a_learning(
        env,
        gamma=args.gamma,
        epsilons=epsilons,
        lr=args.lr,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        n_steps_lim=args.n_steps_lim,
        test_freq_episodes=args.test_freq_episodes,
        seed=args.seed,
        value_function_opt=-sol_hjb.value_function,
        policy_opt=sol_hjb.u_opt,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plot
    if not args.plot:
        return

    # unpack data
    returns = data['returns']
    avg_returns = data['avg_returns']
    time_steps = data['time_steps']
    avg_time_steps = data['avg_time_steps']
    n_table = data['n_table']
    a_table = data['a_table']

    # compute tables
    #TODO!

    # do plots

    plot_returns_episodes(returns, avg_returns)
    plot_time_steps_episodes(time_steps, avg_time_steps)
    plot_frequency(env, n_table)
    #plot_q_table(env, a_table)
    #plot_v_table(env, q_table, sol_hjb.value_function)
    #plot_greedy_policy(env, a_table, sol_hjb.u_opt)
    plot_advantage_function(env, a_table)


if __name__ == '__main__':
    main()
