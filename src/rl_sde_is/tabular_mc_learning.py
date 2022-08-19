import numpy as np

from sde.langevin_sde import LangevinSDE
from hjb.hjb_solver import SolverHJB

from base_parser import get_base_parser
from environments import DoubleWellStoppingTime1D
from tabular_learning import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def mc_learning(env, gamma=1., epsilons=None, constant_alpha=False, alpha=0.01,
                n_episodes=1000, n_avg_episodes=10, n_steps_lim=1000):

    # initialize frequency and q-value function table
    n_table = np.zeros((env.n_states, env.n_actions))
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set
    q_table[env.idx_lb:env.idx_rb+1] = 0

    # preallocate returns and time steps
    returns = np.empty(n_episodes)
    avg_returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    avg_time_steps = np.empty(n_episodes)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.state_init.copy()

        # reset trajectory
        states = np.empty(0)
        actions = np.empty(0)
        rewards = np.empty(0)

        # terminal state flag
        complete = False

        # get epsilon
        epsilon = epsilons[ep]

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # save state
            states = np.append(states, state)

            # get index of the state
            idx_state = env.get_state_idx(state)

            # choose action following epsilon greedy policy
            idx_action, action = get_epsilon_greedy_action(env, q_table, epsilon, idx_state)

            # step dynamics forward
            new_state, r, complete = env.step(state, action)

            # save action and reward
            rewards = np.append(rewards, r)
            actions = np.append(actions, action)

            # update state
            state = new_state

        # compute returns at each time step
        ep_returns = discount_cumsum(rewards, gamma)

        # update q values
        n_steps_trajectory = states.shape[0]
        for k in np.arange(n_steps_trajectory):

            # state and its index at step k
            state = states[k]
            idx_state = env.get_state_idx(state)

            # action and its index at step k
            action = actions[k]
            idx_action = env.get_action_idx(action)

            # state-action index
            idx = (idx_state, idx_action)
            g = ep_returns[k]

            # update frequency table
            n_table[idx] += 1

            # set learning rate
            if not constant_alpha:
                alpha = 1 / n_table[idx]

            # update q table
            q_table[idx] += alpha * (g - q_table[idx])

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            idx_last_episodes = slice(0, ep + 1)
        else:
            idx_last_episodes = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        returns[ep] = ep_returns[0]
        avg_returns[ep] = np.mean(returns[idx_last_episodes])
        time_steps[ep] = n_steps_trajectory
        avg_time_steps[ep] = np.mean(time_steps[idx_last_episodes])

        # logs
        if ep % n_avg_episodes == 0:
            msg = 'ep: {:3d}, run avg return {:2.2f}, ' \
                    'run avg time steps: {:2.2f}, epsilon: {:.2f}'.format(
                    ep,
                    avg_returns[ep],
                    avg_time_steps[ep],
                    epsilon,
                )
            print(msg)

    return returns, avg_returns, time_steps, avg_time_steps, n_table, q_table



def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get target set indices
    env.get_idx_target_set()

    # set epsilons
    epsilons = get_epsilons_linear_decay(args.n_episodes, args.eps_min)

    # run mc learning algorithm
    info = mc_learning(
        env,
        gamma=args.gamma,
        epsilons=epsilons,
        constant_alpha=args.constant_alpha,
        alpha=args.alpha,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        n_steps_lim=args.n_steps_lim,
    )

    returns, avg_returns, time_steps, avg_time_steps, n_table, q_table = info

    # do plots

    # initialize Langevin sde
    sde = LangevinSDE(
        problem_name='langevin_stop-t',
        potential_name='nd_2well',
        d=1,
        alpha=np.ones(1),
        beta=1.,
        domain=np.full((1, 2), [-2, 2]),
    )

    # load hjb solution
    h_hjb = 0.01
    sol_hjb = SolverHJB(sde, h=h_hjb)
    sol_hjb.load()

    # discretization step ratio
    k = int(args.h_state / h_hjb)
    assert env.state_space_h.shape == sol_hjb.u_opt[::k, 0].shape, ''

    #agent.episodes = np.arange(agent.n_episodes)
    #agent.plot_total_rewards()
    #agent.plot_time_steps()
    #agent.plot_epsilons()
    plot_frequency_table(env, n_table)
    plot_q_table(env, q_table)
    plot_greedy_policy(env, q_table, control_hjb=sol_hjb.u_opt[::k])
    #agent.plot_sliced_q_tables()


if __name__ == '__main__':
    main()
