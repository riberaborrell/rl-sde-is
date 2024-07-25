import gymnasium as gym
import gym_sde_is
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.spg.spg_core import *
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.description = 'Run reinforce stochastic for the sde importance sampling environment'
    parser.add_argument(
        '--algorithm-type',
        choices=['initial-return', 'n-return'],
        default='initial-return',
        help='Set reinforce stoch algorithm type. Default: initial-return',
    )
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        alpha=args.alpha,
        beta=args.beta,
        state_init_dist=args.state_init_dist,
    )

    # discretize state and action space (plot purposes only)
    h_coarse = 0.05
    env.discretize_state_space(h_state=h_coarse)

    # compute corresponding beta
    beta = 2 / (env.dt*args.policy_noise**2 + env.sigma**2)
    sigma = np.sqrt(2 / beta)

    # get hjb solver
    sol_hjb = env.get_hjb_solver(beta=1)
    sol_hjb.coarse_solution(h_coarse)
    policy_opt = sol_hjb.u_opt * sigma

    # run reinforce with initial return
    data = reinforce_stochastic(
        env,
        algorithm_type=args.algorithm_type,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        policy_type=args.policy_type,
        policy_noise=args.policy_noise,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        n_grad_iterations=args.n_grad_iterations,
        backup_freq=args.backup_freq,
        policy_opt=policy_opt,
        load=args.load,
        live_plot_freq=args.live_plot_freq,
    )

    # plot results
    if not args.plot:
        return

    # get backup policies
    iterations = np.arange(0, args.n_iterations + args.backup_freq, args.backup_freq)
    means = get_means(env, data, iterations[::10])

    # plot avg returns and mfht
    """
    x = np.arange(data['n_iterations'])
    plot_y_per_x(x, data['objectives'], title='Objective function', xlabel='Iterations')
    plot_y_per_x(x, data['losses'], title='Effective loss', xlabel='Iterations')
    plot_y_per_x(x, data['loss_vars'], title='Effective loss (variance)', xlabel='Iterations')
    plot_y_per_x(x, data['mfhts'], title='MFHT', xlabel='Iterations')
    """

    # plot policy
    if env.d == 1:
        plot_det_policies_1d(env, means, policy_opt)

if __name__ == '__main__':
    main()
