import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.utils_path import *
from rl_sde_is.approximate_methods import *
from rl_sde_is.plots import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def agent_episodic(env, agent='random', batch_size=10, n_episodes=100, n_steps_lim=100000, seed=1,
                   sol_hjb=None, load=False, plot=False, save_traj=False):

    if not agent in ['random', 'not-controlled', 'hjb']:
        raise ValueError

    # get dir path
    rel_dir_path = get_agent_dir_path(
        env,
        agent=agent,
        n_episodes=n_episodes,
        seed=seed,
    )

    # load results
    if load:
        data = load_data(rel_dir_path)
        return data

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # preallocate returns
    returns = []
    time_steps = []

    # preallocate trajectory
    ep_states = np.empty((0,))
    ep_actions = np.empty((0,))
    ep_rewards = np.empty((0,))

    # sample trajectories
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # terminal state flag
        complete = False

        # initialize episodic return
        ep_return = 0.

        for k in range(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # copy state
            state_copy = state.copy()

            # take a random action
            if agent == 'random':
                action = np.random.rand(1) * env.action_space_high
            elif agent == 'not-controlled':
                action = np.zeros(1)
            elif agent == 'hjb':
                idx = env.get_state_idx(state)
                action = sol_hjb.u_opt[idx]

            # step dynamics forward
            new_state, r, complete = env.step(state, action)

            #print('step: {}, state: {:.1f}, action: {:.1f}, reward: {:.3f}'
            #      ''.format(k, state_copy[0], action[0], r))

            # compute return
            ep_return += r

            # save first trajectory
            if save_traj and ep == 0:
                ep_states = np.append(ep_states, state)
                ep_actions = np.append(ep_actions, action)
                ep_rewards = np.append(ep_rewards, r)

            # update state
            state = new_state

        # save episode info
        returns.append(ep_return)
        time_steps.append(k)

    data = {
        'n_episodes': n_episodes,
        'returns': returns,
        'time_steps': time_steps,
        'ep_states': ep_states,
        'ep_actions': ep_actions,
        'ep_rewards': ep_rewards,
    }
    save_data(data, rel_dir_path)
    return data


def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D(
        alpha=args.alpha,
        beta=args.beta,
        dt=args.dt,
    )

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize state space (for plot purposes only)
    env.discretize_state_space(h_state=0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run agent with random actions
    data = agent_episodic(
        env,
        agent='not-controlled',
        batch_size=args.batch_size,
        n_episodes=args.n_episodes,
        save_traj=args.save_traj,
        sol_hjb=sol_hjb,
        load=args.load,
        plot=args.plot,
    )

    # plots
    if not args.plot:
        return

    # plot trajectory
    ep_states = data['ep_states']
    if ep_states.shape[0] != 0:
        plot_episode_states(env, ep_states)
    return

    # smoothed arrays
    returns = data['returns']
    time_steps = data['time_steps']
    run_mean_returns = compute_running_mean(returns, args.batch_size)
    run_var_returns = compute_running_variance(returns, args.batch_size)
    run_mean_time_steps = compute_running_mean(time_steps, args.batch_size)
    plot_returns_episodes(returns, run_mean_returns)
    plot_time_steps_episodes(time_steps, run_mean_time_steps)

if __name__ == '__main__':
    main()
