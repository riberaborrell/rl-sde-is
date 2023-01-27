# first-visit mc prediction (sutton and barto)

import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.tabular_learning import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def plot_value_function(env, v_table, value_f_hjb):

    # discretize state space
    x = env.state_space_h

    fig, ax = plt.subplots()
    ax.plot(x, -v_table)
    ax.plot(x, value_f_hjb)
    #fig.set_ylim(-100, 0)
    plt.show()


def mc_prediction(env, policy, gamma=1.0, n_episodes=100, n_avg_episodes=10,
                  n_steps_lim=1000, first_visit=False):
    '''
    '''
    # initialize value function table and returns
    v_table = -np.random.rand(env.n_states)
    returns_table = [[] for i in range(env.n_states)]

    # set values for the target set
    v_table[env.idx_lb:env.idx_rb+1] = 0

    # preallocate returns and time steps
    returns = np.empty(n_episodes)
    avg_returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    avg_time_steps = np.empty(n_episodes)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # reset trajectory
        states = np.empty(0)
        rewards = np.empty(0)

        idx_states = []

        # terminal state flag
        done = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # save state
            states = np.append(states, state)

            # get index of the state
            idx_state = env.get_state_idx(state)
            idx_states.append(idx_state)

            # choose action following the given policy
            idx_action = policy[idx_state]
            action = env.action_space_h[[idx_action]]

            # step dynamics forward
            new_state, r, done = env.step(state, action)

            # save reward
            rewards = np.append(rewards, r)

            # update state
            state = new_state

        # save number of time steps of the episode
        time_steps[ep] = k

        ret = 0
        for k in np.flip(np.arange(time_steps[ep])):

            # compute return at time step k  
            ret = gamma * ret + r

            # if state not in the previous time steps
            if not idx_states[k] in idx_states[:k] or not first_visit:

                # get current state index
                idx_state = idx_states[k]

                # append return
                returns_table[idx_state].append(ret)

                # update v table
                v_table[idx_state] = np.mean(returns_table[idx_state])

        # save episode return
        returns[ep] = ret

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            idx_last_episodes = slice(0, ep + 1)
        else:
            idx_last_episodes = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        avg_returns[ep] = np.mean(returns[idx_last_episodes])
        avg_time_steps[ep] = np.mean(time_steps[idx_last_episodes])

        # logs
        if ep % n_avg_episodes == 0:
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                  'run avg time steps: {:2.2f}'.format(
                    ep,
                    v_table[env.idx_state_init],
                    avg_returns[ep],
                    avg_time_steps[ep],
                )
            print(msg)

    return returns, avg_returns, time_steps, avg_time_steps, v_table

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # set deterministic policy from the hjb control
    policy = np.array([
        env.get_action_idx(sol_hjb.u_opt[idx_state])
        for idx_state, _ in enumerate(env.state_space_h)
    ])

    # run mc learning agent following optimal policy
    info = mc_prediction(
        env,
        policy,
        gamma=args.gamma,
        n_steps_lim=args.n_steps_lim,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
    )

    returns, avg_returns, time_steps, avg_time_steps, v_table = info

    plot_value_function(env, v_table, value_f_hjb=sol_hjb.value_function)



if __name__ == '__main__':
    main()
