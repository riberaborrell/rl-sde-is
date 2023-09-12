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

def sarsa(env, gamma=1., epsilons=None, lr=0.01, n_episodes=1000,
          n_steps_lim=1000, test_freq_episodes=10, n_avg_episodes=10, seed=None,
          value_function_opt=None, policy_opt=None, load=None, live_plot=False):

    ''' Sarsa
    '''

    # get dir path
    rel_dir_path = get_sarsa_lambda_dir_path(
        env,
        agent='tabular-sarsa',
        eps_type='constant',
        eps_init=0,
        n_episodes=n_episodes,
        lr=lr,
        lam=1,
        seed=seed,
    )

    # load results
    if load:
        return load_data(rel_dir_path)

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # initialize q-value function table
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set
    q_table[env.ts_idx] = 0

    # initialize frequency table
    n_table = np.zeros((env.n_states, env.n_actions))

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
        v_table, a_table, policy = compute_tables(env, q_table)
        lines = initialize_tabular_figures(env, q_table, v_table, a_table, policy,
                                              value_function_opt, policy_opt)
        im_n_table = initialize_frequency_figure(env, n_table)

    # for each episode
    for ep in np.arange(n_episodes):

        # get epsilon
        epsilon = epsilons[ep]

        # reset environment
        state = env.reset()

        # choose action
        state_idx = env.get_state_idx(state)
        action_idx, action = get_epsilon_greedy_action(env, q_table, state_idx, epsilon)
        idx = (state_idx, action_idx,)

        # reset trajectory rewards
        rewards = np.empty(0)

        # terminal state flag
        done = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # step dynamics forward
            next_state, r, done, _ = env.step(state, action)
            next_state_idx = env.get_state_idx(next_state)

            # get new action
            next_action_idx, next_action \
                    = get_epsilon_greedy_action(env, q_table, next_state_idx, epsilon)
            next_idx = (next_state_idx, next_action_idx,)

            # update frequency table
            n_table[idx] += 1

            # update q-values
            d = np.where(done, 1., 0.)
            target = r + gamma * (1 - d) * q_table[next_idx]
            q_table[idx] += lr * (target - q_table[idx])

            # save reward
            rewards = np.append(rewards, r)

            # update state
            state = next_state

            # update state and action
            idx = next_idx
            state = next_state
            action = next_action

        # compute returns at each time step
        ep_returns = discount_cumsum(rewards, gamma)

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            idx_last_episodes = slice(0, ep + 1)
        else:
            idx_last_episodes = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        returns[ep] = ep_returns[0]
        avg_returns[ep] = np.mean(returns[idx_last_episodes])
        time_steps[ep] = rewards.shape[0]
        avg_time_steps[ep] = np.mean(time_steps[idx_last_episodes])

        # test
        if (ep + 1) % test_freq_episodes == 0:

            # compute root mean square error of value function and control
            ep_test = (ep + 1) // test_freq_episodes
            v_table, a_table, policy = compute_tables(env, q_table)
            v_rms_errors[ep_test] = compute_rms_error(value_function_opt, v_table)
            p_rms_errors[ep_test] = compute_rms_error(policy_opt, policy)

            # logs
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                    'run avg time steps: {:2.2f}, epsilon: {:.2f}'.format(
                    ep,
                    np.max(q_table[env.state_init_idx]),
                    avg_returns[ep],
                    avg_time_steps[ep],
                    epsilon,
                )
            print(msg)

            # update live figures
            if live_plot:
                update_tabular_figures(env, q_table, v_table, a_table, policy, lines)
                update_frequency_figure(env, n_table, im_n_table)

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
        'q_table' : q_table,
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

    # set action space bounds
    env.action_space_low = -5
    env.action_space_high = 5

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # set epsilons
    epsilons = get_epsilons_constant(args.n_episodes, eps_init=0.01)
    #epsilons = get_epsilons_constant(args.n_episodes, eps_init=1.)
    #epsilons = get_epsilons_linear_decay(args.n_episodes, eps_min=0.01, exploration=0.5)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run sarsa algorithm
    data = sarsa(
        env,
        gamma=args.gamma,
        epsilons=epsilons,
        lr=args.lr,
        n_episodes=args.n_episodes,
        test_freq_episodes=args.test_freq_episodes,
        n_steps_lim=args.n_steps_lim,
        seed=args.seed,
        value_function_opt=-sol_hjb.value_function,
        policy_opt=sol_hjb.u_opt,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plot
    if not args.plot:
        return

    # compute tables
    q_table = data['q_table']
    v_table, a_table, policy_greedy = compute_tables(env, q_table)

    # do plots
    plot_value_function_1d(env, v_table, -sol_hjb.value_function)
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table)
    plot_det_policy_1d(env, policy_greedy, sol_hjb.u_opt)
    plot_value_rms_error_episodes(data['v_rms_errors'], data['test_freq_episodes'])
    plot_policy_rms_error_episodes(data['p_rms_errors'], data['test_freq_episodes'])


if __name__ == '__main__':
    main()
