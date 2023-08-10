import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import *
from rl_sde_is.utils_path import *
from rl_sde_is.utils_numeric import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def double_q_learning(env, gamma=1., epsilons=None, eps_init=None, eps_decay=None, lr=0.01,
                      n_episodes=1000, n_avg_episodes=10, n_steps_lim=1000, seed=None,
                      value_function_opt=None, policy_opt=None, load=False, live_plot=False):

    # get dir path
    rel_dir_path = get_qlearning_dir_path(
        env,
        agent='tabular-double-qlearning',
        n_episodes=n_episodes,
        lr=lr,
        seed=seed,
        eps_type='exp-decay',
        eps_init=eps_init,
        eps_decay=eps_decay,
    )

    # load results
    if load:
        return load_data(rel_dir_path)

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # initialize frequency and q-value function table
    n_table = np.zeros((env.n_states, env.n_actions), dtype=np.int32)
    q1_table = - np.random.rand(env.n_states, env.n_actions)
    q2_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set
    q1_table[env.ts_idx] = 0
    q2_table[env.ts_idx] = 0

    # preallocate returns and time steps
    returns = np.empty(n_episodes)
    avg_returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    avg_time_steps = np.empty(n_episodes)

    # preallocate value function and control rms errors
    v_rms_errors = np.empty(n_episodes)
    p_rms_errors = np.empty(n_episodes)

    # initialize live figures
    if live_plot:
        v_table, a_table, policy = compute_tables(env, q1_table)
        im = initialize_frequency_figure(env, n_table)
        lines = initialize_q_learning_figures(env, q1_table, v_table, a_table, policy,
                                              value_function_opt, policy_opt)
    # for each episode
    for ep in np.arange(n_episodes):

        # get epsilon
        epsilon = epsilons[ep]

        # reset environment
        state = env.reset()

        # reset trajectory rewards
        rewards = np.empty(0)

        # terminal state flag
        done = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # choose action
            state_idx = env.get_state_idx(state)
            action_idx, action = get_epsilon_greedy_action(env, q1_table + q2_table,
                                                           state_idx, epsilon)

            # update frequency table
            idx = (state_idx, action_idx,)
            n_table[idx] += 1

            # step dynamics forward
            next_state, r, done, _ = env.step(state, action)
            next_state_idx = env.get_state_idx(next_state)

            d = np.where(done, 1., 0.)

            # update q values
            if np.random.rand() > 0.5:
                target = r + gamma * (1 - d) * q2_table[next_state_idx, np.argmax(q1_table[next_state_idx])]
                q1_table[idx] += lr * (target - q1_table[idx])
            else:
                target = r + gamma * (1 - d) * q1_table[next_state_idx, np.argmax(q2_table[next_state_idx])]
                q2_table[idx] += lr * (target - q2_table[idx])

            # save action and reward
            rewards = np.append(rewards, r)

            # update state
            state = next_state

        # compute return
        ep_returns = discount_cumsum(rewards, gamma)

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            last_episodes_idx = slice(0, ep + 1)
        else:
            last_episodes_idx = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        returns[ep] = ep_returns[0]
        avg_returns[ep] = np.mean(returns[last_episodes_idx])
        time_steps[ep] = rewards.shape[0]
        avg_time_steps[ep] = np.mean(time_steps[last_episodes_idx])

        # compute root mean square error of value function and control
        v_table, a_table, policy = compute_tables(env, q1_table)
        v_rms_errors[ep] = compute_rms_error(value_function_opt, v_table)
        p_rms_errors[ep] = compute_rms_error(policy_opt, policy)

        # logs
        if ep % n_avg_episodes == 0:
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                  'run avg time steps: {:2.2f}, epsilon: {:2.3f}'.format(
                    ep,
                    np.max(q1_table[env.state_init_idx]),
                    avg_returns[ep],
                    avg_time_steps[ep],
                    epsilon,
                )
            print(msg)

        # update live figures
        if live_plot and ep % n_avg_episodes == 0:
            v_table, a_table, policy = compute_tables(env, q1_table)
            update_frequency_figure(env, n_table, im)
            update_q_learning_figures(env, q1_table, v_table, a_table, policy, lines)

    data = {
        'n_episodes': n_episodes,
        'returns': returns,
        'avg_returns': avg_returns,
        'time_steps': time_steps,
        'avg_time_steps': avg_time_steps,
        'n_table' : n_table,
        'q1_table' : q1_table,
        'q2_table' : q2_table,
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
    #epsilons = get_epsilons_constant(args.n_episodes, eps_init=args.eps_init)
    #epsilons = get_epsilons_linear_decay(args.n_episodes, eps_min=0.01, exploration=0.5)
    epsilons = get_epsilons_exp_decay(args.n_episodes, eps_init=args.eps_init, eps_decay=args.decay)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run double q learning algorithm
    data = double_q_learning(
        env,
        gamma=args.gamma,
        epsilons=epsilons,
        eps_init=args.eps_init,
        eps_decay=args.decay,
        lr=args.lr,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
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
    q1_table, q2_table = data['q1_table'], data['q2_table']
    q_table = (q1_table + q2_table) / 2
    v_table, a_table, policy_greedy = compute_tables(env, q_table)

    # do plots
    plot_value_function_1d(env, v_table, sol_hjb.value_function)
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table)
    plot_det_policy_1d(env, policy_greedy, sol_hjb.u_opt)
    plot_value_rms_error_episodes(data['v_rms_errors'])
    plot_policy_rms_error_epochs(data['p_rms_errors'])

    # do plots
    plot_returns_episodes(data['returns'], data['avg_returns'])
    plot_time_steps_episodes(data['time_steps'], data['avg_time_steps'])
    plot_frequency(env, n_table)


if __name__ == '__main__':
    main()