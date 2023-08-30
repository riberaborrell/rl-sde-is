import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.environments_2d import DoubleWellStoppingTime2D
from rl_sde_is.utils_path import *
from rl_sde_is.approximate_methods import *
from rl_sde_is.tabular_methods import evaluate_policy_vectorized
from rl_sde_is.plots import *
from rl_sde_is.utils_numeric import compute_running_mean, compute_running_variance

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def policy_evaluation(env, agent, batch_size=1000, n_steps_lim=100000, seed=1,
                      sol_hjb=None, load=False):

    if not agent in ['not-controlled', 'hjb']:
        raise ValueError

    # get dir path
    rel_dir_path = get_agent_dir_path(
        env,
        agent=agent,
        batch_size=batch_size,
        seed=seed,
    )

    # load results
    if load:
        return load_data(rel_dir_path)

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # set not controlled policy
    if agent == 'not-controlled':
        policy = np.repeat(env.null_action_idx, env.n_states)

    # set deterministic policy from the hjb control
    else:
        policy = env.get_det_policy_indices_from_hjb(sol_hjb.u_opt)

    # evaluate policy
    returns, time_steps = evaluate_policy_vectorized(env, policy, batch_size)

    data = {
        'batch_size': batch_size,
        'returns': returns,
        'time_steps': time_steps,
        #'ep_states': ep_states,
        #'ep_actions': ep_actions,
        #'ep_rewards': ep_rewards,
    }
    save_data(data, rel_dir_path)
    return data


def main():
    args = get_parser().parse_args()

    # initialize environment
    if args.d == 1:
        env = DoubleWellStoppingTime1D(
            alpha=args.alpha,
            beta=args.beta,
            dt=args.dt,
        )
    #elif args.d == 2:
    #    env = DoubleWellStoppingTime2D(
    #        alpha=args.alpha,
    #        beta=args.beta,
    #        dt=args.dt,
    #    )
    else:
        #raise ValueError('just case 1d and 2d covered')
        raise ValueError('just case 1d covered')

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize state space and action space
    env.set_action_space_bounds()
    env.discretize_state_space(0.01)
    env.discretize_action_space(0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run agent
    data = policy_evaluation(
        env,
        agent=args.agent_type,
        batch_size=args.batch_size,
        #save_traj=args.save_traj,
        sol_hjb=sol_hjb,
        load=args.load,
    )

    # plots
    if not args.plot:
        return

    # plot trajectory
    #ep_states = data['ep_states']
    #if args.d == 1 and ep_states.shape[0] != 0:
    #    plot_episode_states_1d(env, ep_states)
    #elif args.d == 2 and ep_states.shape[0] != 0:
    #    plot_episode_states_2d(env, ep_states)

    # smoothed arrays
    returns = data['returns']
    time_steps = data['time_steps']
    plot_time_steps_histogram(time_steps)
    run_mean_returns = compute_running_mean(returns, 10)
    run_var_returns = compute_running_variance(returns, 10)
    run_mean_time_steps = compute_running_mean(time_steps, 10)
    plot_returns_episodes(returns, run_mean_returns)
    plot_time_steps_episodes(time_steps, run_mean_time_steps)

if __name__ == '__main__':
    main()
