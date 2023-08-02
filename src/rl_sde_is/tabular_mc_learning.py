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

def mc_learning(env, gamma=1., epsilons=None, constant_lr=False, lr=0.01, n_episodes=1000,
                n_steps_lim=1000, test_freq_episodes=10, n_avg_episodes=10, seed=None,
                value_function_opt=None, policy_opt=None, load=None, live_plot=False):

    ''' Mc learning
    '''

    # get dir path
    rel_dir_path = get_mc_learning_dir_path(
        env,
        agent='mc-learning',
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
        lines = initialize_q_learning_figures(env, q_table, v_table, a_table, policy,
                                              value_function_opt, policy_opt)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # reset trajectory
        states = np.empty(0)
        actions = np.empty(0)
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

            # save state
            states = np.append(states, state)

            # get index of the state
            state_idx = env.get_state_idx(state)

            # choose action following epsilon greedy policy
            _, action = get_epsilon_greedy_action(env, q_table, state_idx, epsilon)

            # step dynamics forward
            new_state, r, done, _ = env.step(state, action)

            # save action and reward
            rewards = np.append(rewards, r)
            actions = np.append(actions, action)

            # update state
            state = new_state

        # initialize frequency table
        #n_table = np.zeros((env.n_states, env.n_actions))
        n_table = np.zeros(env.n_states)

        # compute returns at each time step
        ep_returns = discount_cumsum(rewards, gamma)

        # update q values
        n_steps_trajectory = states.shape[0]
        for k in np.arange(n_steps_trajectory):

            # state and its index at step k
            state = states[k]
            state_idx = env.get_state_idx(state)

            # action and its index at step k
            action = actions[k]
            action_idx = env.get_action_idx(action)

            # state-action index
            idx = (state_idx, action_idx)
            g = ep_returns[k]

            # update frequency table
            n_table[state_idx] += 1

            # set learning rate
            #if not constant_lr:
            #    lr = 1 / n_table[idx]

            # update q table
            q_table[idx] += (lr / n_table[state_idx]) * (g - q_table[idx])

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            last_episodes_idx = slice(0, ep + 1)
        else:
            last_episodes_idx = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        returns[ep] = ep_returns[0]
        avg_returns[ep] = np.mean(returns[last_episodes_idx])
        time_steps[ep] = n_steps_trajectory
        avg_time_steps[ep] = np.mean(time_steps[last_episodes_idx])


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
                    np.max(q_table[env.state_init_idx.item()]),
                    avg_returns[ep],
                    avg_time_steps[ep],
                    epsilon,
                )
            print(msg)

            # update live figures
            if live_plot:
                update_q_learning_figures(env, q_table, v_table, a_table, policy, lines)

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
    env.set_action_space_bounds()

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # set epsilons
    epsilons = get_epsilons_constant(args.n_episodes, eps_init=0.01)
    #epsilons = get_epsilons_constant(args.n_episodes, eps_init=1.)
    #epsilons = get_epsilons_linear_decay(args.n_episodes, eps_min=0.01, exploration=0.5)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run mc learning algorithm
    data = mc_learning(
        env,
        gamma=args.gamma,
        epsilons=epsilons,
        constant_lr=args.constant_lr,
        lr=args.lr,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
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
