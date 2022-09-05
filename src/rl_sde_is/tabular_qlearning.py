import numpy as np

from base_parser import get_base_parser
from environments import DoubleWellStoppingTime1D
from tabular_learning import *
from utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def q_learning(env, gamma=1., lr=0.01, n_episodes=1000, n_avg_episodes=10, n_steps_lim=1000,
               eps_type='linear-decay', eps_init=1., eps_min=0., eps_decay=0.98,
               value_function_hjb=None, control_hjb=None, load=False):

    # get dir path
    dir_path = get_qlearning_dir_path(
        env,
        lr=lr,
        eps_type=eps_type,
        eps_min=eps_min,
        n_episodes=n_episodes,
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
    #epsilons = get_epsilons_constant(n_episodes, eps_init)
    #epsilons = get_epsilons_harmonic(n_episodes)
    epsilons = get_epsilons_linear_decay(n_episodes, eps_min, exploration=0.5)
    #epsilons = get_epsilons_exp_decay(n_episodes, eps_init, eps_decay)

    # initialize plots 
    #images, lines = initialize_figures(env, n_table, q_table, n_episodes,
    #                                   value_function_hjb, control_hjb)

    # preallocate returns and time steps
    returns = np.empty(n_episodes)
    avg_returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    avg_time_steps = np.empty(n_episodes)

    # fill with nan values (for plotting purposes only)
    returns.fill(np.nan)
    avg_returns.fill(np.nan)
    time_steps.fill(np.nan)
    avg_time_steps.fill(np.nan)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # reset trajectory rewards
        rewards = np.empty(0)

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # get index of the state
            idx_state = env.get_state_idx(state)

            # get epsilon
            epsilon = epsilons[ep]

            # choose action following epsilon greedy action
            idx_action, action = get_epsilon_greedy_action(env, q_table, idx_state, epsilon)

            # get idx state-action pair
            idx = (idx_state, idx_action,)

            # step dynamics forward
            new_state, r, complete = env.step(state, action)
            idx_new_state = env.get_state_idx(new_state)

            # update q values
            n_table[idx] += 1
            q_table[idx] += lr * (
                  r \
                + gamma * np.max(q_table[idx_new_state]) \
                - q_table[idx]
            )

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

        # update plots
        #update_figures(env, n_table, q_table, returns, avg_returns, time_steps,
        #               avg_time_steps, images, lines)

        # logs
        if ep % n_avg_episodes == 0:
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                  'run avg time steps: {:2.2f}, epsilon: {:2.3f}'.format(
                    ep,
                    np.max(q_table[env.idx_state_init]),
                    avg_returns[ep],
                    avg_time_steps[ep],
                    epsilon,
                )
            print(msg)

    data = {
        'n_episodes': n_episodes,
        'returns': returns,
        'avg_returns': avg_returns,
        'time_steps': time_steps,
        'avg_time_steps': avg_time_steps,
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
        lr=args.lr,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        n_steps_lim=args.n_steps_lim,
        eps_type=args.eps_type,
        eps_init=args.eps_init,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
    )

    #agent.episodes = np.arange(agent.n_episodes)
    #agent.plot_total_rewards()
    #agent.plot_time_steps()
    #agent.plot_epsilons()
    plot_frequency_table(env, data['n_table'])
    plot_q_table(env, data['q_table'])
    plot_v_table(env, data['q_table'], sol_hjb.value_function)
    plot_a_table(env, data['q_table'])
    plot_greedy_policy(env, data['q_table'], sol_hjb.u_opt)


if __name__ == '__main__':
    main()
