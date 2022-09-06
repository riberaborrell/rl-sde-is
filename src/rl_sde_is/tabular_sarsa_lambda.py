import numpy as np
import matplotlib.pyplot as plt

from base_parser import get_base_parser
from environments import DoubleWellStoppingTime1D
from tabular_learning import *
from utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def sarsa_learning(env, gamma=1., lr=0.01, lam=0.1, n_episodes=1000, n_avg_episodes=10,
                   eps_type='linear-decay', eps_init=1., eps_min=0., eps_decay=0.98,
                   n_steps_lim=1000, load=False):

    # get dir path
    dir_path = get_sarsa_lambda_dir_path(
        env,
        lr=lr,
        lam=lam,
        n_episodes=n_episodes,
        eps_type=eps_type,
        eps_init=eps_init,
        eps_min=eps_min,
        eps_decay=eps_decay,
    )

    # load results
    if load:
        data = load_data(dir_path)
        return data


    # initialize frequency, q-value and eligibility traces tables
    n_table = np.zeros((env.n_states, env.n_actions), dtype=np.int32)
    q_table = - np.random.rand(env.n_states, env.n_actions)
    e_table = np.zeros((env.n_states, env.n_actions))

    # set values for the target set
    q_table[env.idx_lb:env.idx_rb+1] = 0

    # set epsilons
    epsilons = set_epsilons(
        eps_type,
        n_episodes=n_episodes,
        eps_init=eps_init,
        eps_min=eps_min,
        eps_decay=eps_decay,
    )

    # preallocate returns and time steps
    returns = np.empty(n_episodes)
    avg_returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    avg_time_steps = np.empty(n_episodes)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # get index of the state
        idx_state = env.get_state_idx(state)

        # get epsilon
        epsilon = epsilons[ep]

        # choose action following epsilon greedy action
        idx_action, action = get_epsilon_greedy_action(env, q_table, idx_state, epsilon)

        # get idx state-action pair
        idx = (idx_state, idx_action,)

        # reset trajectory rewards
        rewards = np.empty(0)

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # step dynamics forward
            new_state, r, complete = env.step(state, action)
            idx_new_state = env.get_state_idx(new_state)

            # choose action following epsilon greedy action
            idx_new_action, new_action = get_epsilon_greedy_action(env, q_table, idx_new_state, epsilon)
            idx_new = (idx_new_state, idx_new_action,)

            # compute td error
            td_error = r + gamma * q_table[idx_new] - q_table[idx]

            # update eligibility traces table and frequency table
            e_table[idx] += 1
            n_table[idx] += 1

            # update q table and eligibility traces table for all states and actions
            q_table += lr * td_error * e_table
            e_table *= gamma * lam

            # save reward
            rewards = np.append(rewards, r)

            # update state and action
            state = new_state
            action = new_action
            idx = idx_new

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

        # logs
        if ep % n_avg_episodes == 0:
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                    'run avg time steps: {:2.2f}, epsilon: {:.2f}'.format(
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


def plot_frequency_table(env, n_table):
    # set extent bounds
    extent = env.state_space_h[0], env.state_space_h[-1], \
             env.action_space_h[0], env.action_space_h[-1]

    fig, ax = plt.subplots()

    im = fig.axes[0].imshow(
        n_table.T,
        origin='lower',
        extent=extent,
        cmap=cm.coolwarm,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_q_table(env, q_table):
    # set extent bounds
    extent = env.state_space_h[0], env.state_space_h[-1], \
             env.action_space_h[0], env.action_space_h[-1]

    fig, ax = plt.subplots()

    im = fig.axes[0].imshow(
        q_table.T,
        origin='lower',
        extent=extent,
        cmap=cm.viridis,
    )

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plot_greedy_policy(env, q_table, control_hjb):

    x = env.state_space_h

    # compute greedy policy by following the q-table
    greedy_policy = np.empty_like(x)
    for idx, x_k in enumerate(x):
        idx_action = np.argmax(q_table[idx])
        greedy_policy[idx] = env.action_space_h[idx_action]

    fig, ax = plt.subplots()
    plt.plot(x, greedy_policy)
    plt.plot(x, control_hjb[:, 0])
    plt.show()

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

    # run sarsa learning algorithm
    data = sarsa_learning(
        env,
        gamma=args.gamma,
        lr=args.lr,
        lam=args.lam,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        n_steps_lim=args.n_steps_lim,
        eps_type=args.eps_type,
        eps_init=args.eps_init,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        load=args.load,
    )

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

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
