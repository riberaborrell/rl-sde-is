import numpy as np

from base_parser import get_base_parser
from environments import DoubleWellStoppingTime1D
from tabular_learning import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def plot_given_policy(env, policy, control_hjb):

    # discretize state space
    x = env.state_space_h

    # get actions following given policy
    actions = np.array([
        env.action_space_h[idx_action] for idx_action in policy
    ])

    fig, ax = plt.subplots()
    ax.plot(x, actions)
    ax.plot(x, control_hjb[:, 0])
    plt.show()

def plot_value_function(env, v_table, value_f_hjb):

    # discretize state space
    x = env.state_space_h

    fig, ax = plt.subplots()
    ax.plot(x, -v_table)
    ax.plot(x, value_f_hjb)
    #fig.set_ylim(-100, 0)
    plt.show()


def mc_prediction(env, policy, gamma=1.0, n_episodes=100, n_avg_episodes=10,
                  n_steps_lim=1000, constant_alpha=True, alpha=0.01):
    '''
    '''
    # preallocate alphas if they are not constant
    if not constant_alpha:
        alphas = np.empty(0)

    # initialize frequency and value function table
    n_table = np.zeros(env.n_states)
    v_table = -np.random.rand(env.n_states)

    # set values for the target set
    v_table[env.idx_lb:env.idx_rb+1] = 0

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init)

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
        rewards = np.empty(0)

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # save state
            states = np.append(states, state)

            # get index of the state
            idx_state = env.get_state_idx(state)

            # choose action following the given policy
            idx_action = policy[idx_state]
            action = env.action_space_h[[idx_action]]

            # step dynamics forward
            new_state, r, complete = env.step(state, action)

            # save reward
            rewards = np.append(rewards, r)

            # update state
            state = new_state

        # compute the returns along the trajectory
        ep_returns = discount_cumsum(rewards, gamma)

        # update v values
        n_steps_trajectory = states.shape[0]
        for k in np.arange(n_steps_trajectory):

            # state and its index at step k
            state = states[k]
            idx_state = env.get_state_idx(state)

            # return at time step k
            g = ep_returns[k]

            # update frequency table
            n_table[idx_state] += 1

            # set learning rate
            if not constant_alpha:
                alpha = 1 / n_table[idx_state]
                alphas = np.append(alphas, alpha)

            # update v table
            v_table[idx_state] += alpha * (g - v_table[idx_state])

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
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                  'run avg time steps: {:2.2f}'.format(
                    ep,
                    v_table[idx_state_init],
                    avg_returns[ep],
                    avg_time_steps[ep],
                )
            print(msg)

    return returns, avg_returns, time_steps, avg_time_steps, n_table, v_table

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get target set indices
    env.get_idx_target_set()

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # factor between the two different discretizations steps
    k = int(args.h_state / sol_hjb.sde.h)
    assert env.state_space_h.shape == sol_hjb.u_opt[::k, 0].shape, ''

    # set deterministic policy from the hjb control
    policy = np.array([
        env.get_action_idx(sol_hjb.u_opt[::k][idx_state])
        for idx_state, _ in enumerate(env.state_space_h)
    ])

    # run mc learning agent following optimal policy
    info = mc_prediction(
        env,
        policy,
        gamma=args.gamma,
        constant_alpha=args.constant_alpha,
        alpha=args.alpha,
        n_steps_lim=args.n_steps_lim,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
    )

    returns, avg_returns, time_steps, avg_time_steps, n_table, v_table = info

    #plot_given_policy(env, policy, control_hjb=sol_hjb.u_opt[::k])
    plot_value_function(env, v_table, value_f_hjb=sol_hjb.value_function[::k])
    #if not agent.constant_alpha:
    #    agent.plot_alphas()



if __name__ == '__main__':
    main()
