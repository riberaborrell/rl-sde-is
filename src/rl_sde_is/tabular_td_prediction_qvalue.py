import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def compute_trajectory_return(rewards, gamma):
    n = rewards.shape[0]
    gammas = gamma**np.arange(n)
    return sum(rewards * gammas)

def get_epsilon_greedy_action(env, policy, idx_state, epsilon):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        idx_action = policy[idx_state]

    # pick random action (exploration)
    else:
        idx_action = np.random.choice(np.arange(env.n_actions))

    action = env.action_space_h[[idx_action]]

    return idx_action, action

def sarsa_prediction(env, policy, gamma=1.0, epsilons=None, n_episodes=100, n_avg_episodes=10,
                  n_steps_lim=1000, alpha=0.01):
    '''
    '''

    # initialize value function table
    q_table = - np.random.rand(env.n_states, env.n_actions)

    # set values for the target set
    q_table[env.idx_lb:env.idx_rb+1] = 0

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
        state = env.reset()

        # get index of the state
        idx_state = env.get_state_idx(state)

        # choose action following the given policy
        idx_action = policy[idx_state]
        action = env.action_space_h[idx_action]
        #idx_action, action = get_epsilon_greedy_action(env, policy, idx_state, epsilons[ep])

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
            new_state, r, complete, _ = env.step(state, action)
            idx_new_state = env.get_state_idx(new_state)

            # choose next action following the given policy
            idx_new_action = policy[idx_new_state]
            new_action = env.action_space_h[idx_new_action]

            # update v values
            q_table[idx_state, idx_action] += alpha * (
                #r + gamma * np.max(q_table[idx_new_state]) - q_table[idx_state, idx_action]
                r + gamma * np.max(q_table[idx_new_state, idx_new_action]) - q_table[idx_state, idx_action]
            )

            # save reward
            rewards = np.append(rewards, r)

            # update state and action
            state = new_state
            idx_state = idx_new_state
            action = new_action
            idx_action = idx_new_action

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            idx_last_episodes = slice(0, ep + 1)
        else:
            idx_last_episodes = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        returns[ep] = compute_trajectory_return(rewards, gamma)
        avg_returns[ep] = np.mean(returns[idx_last_episodes])
        time_steps[ep] = k
        avg_time_steps[ep] = np.mean(time_steps[idx_last_episodes])

        # logs
        if ep % n_avg_episodes == 0:
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                  'run avg time steps: {:2.2f}'.format(
                    ep,
                    np.max(q_table[env.idx_state_init]),
                    avg_returns[ep],
                    avg_time_steps[ep],
                )
            print(msg)

    return returns, avg_returns, time_steps, avg_time_steps, q_table

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

    # set deterministic policy from the hjb control
    policy = np.array([
        env.get_action_idx(sol_hjb.u_opt[idx_state])
        for idx_state, _ in enumerate(env.state_space_h)
    ])

    # set epsilons
    #epsilons = get_epsilons_constant(args.n_episodes, eps_init=1.)
    #epsilons = get_epsilons_harmonic(args.n_episodes)
    epsilons = get_epsilons_linear_decay(args.n_episodes, args.eps_min, exploration=0.5)
    #epsilons = get_epsilons_exp_decay(args.n_episodes, args.eps_init, args.eps_decay)

    # run temporal difference learning agent following optimal policy
    info = sarsa_prediction(
        env,
        policy,
        gamma=args.gamma,
        alpha=args.alpha,
        n_steps_lim=args.n_steps_lim,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        epsilons=epsilons,
    )

    returns, avg_returns, time_steps, avg_time_steps, q_table = info

    # compute data
    v_table, a_table, policy_greedy = compute_tables(env, q_table)

    # do plots
    plot_value_function_1d(env, v_table, sol_hjb.value_function)
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table)
    plot_det_policy_1d(env, policy_greedy, sol_hjb.u_opt)


if __name__ == '__main__':
    main()
