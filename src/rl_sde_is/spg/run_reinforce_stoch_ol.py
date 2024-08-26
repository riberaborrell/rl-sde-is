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
        '--return-type',
        choices=['initial-return', 'n-return'],
        default='initial-return',
        help='Set type of return used. Default: initial-return',
    )
    parser.add_argument(
        '--expectation-type',
        choices=['random-time', 'on-policy', 'off-policy'],
        default='random-time',
        help='Set type of expectation. Default: random-time',
    )
    parser.add_argument(
        '--mini-batch-size',
        type=int,
        default=None,
        help='Set mini batch size for on-policy expectations. Default: None',
    )
    parser.add_argument(
        '--estimate-mfht',
        action='store_true',
        help='Estimate the mfht in the dpg.',
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
    env.discretize_state_space(h_state=args.h_state)

    # get hjb solver
    sol_hjb = env.get_hjb_solver(args.h_state)

    # run reinforce with gaussian stochastic policy
    data = reinforce_stochastic(
        env,
        expectation_type=args.expectation_type,
        return_type=args.return_type,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        theta_init=args.theta_init,
        policy_type=args.gaussian_policy_type,
        policy_noise=args.policy_noise,
        estimate_mfht=args.estimate_mfht,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        memory_size=args.replay_size,
        lr=args.lr,
        seed=args.seed,
        n_grad_iterations=args.n_grad_iterations,
        backup_freq=args.backup_freq,
        policy_opt=sol_hjb.u_opt,
        load=args.load,
        live_plot_freq=args.live_plot_freq,
    )

    # plot results
    if not args.plot:
        return

    # plot statistics
    x = np.arange(data['n_grad_iterations']+1)
    plot_y_per_grad_iteration(x, data['mean_returns'], title='Objective function')
    plot_y_per_grad_iteration(x, data['losses'], title='Effective loss')
    plot_y_per_grad_iteration(x, data['loss_vars'], title='Effective loss (variance)')
    plot_y_per_grad_iteration(x, data['mean_fhts'], title='MFHT')

    # get backup policies
    iterations = np.arange(0, args.n_grad_iterations + args.backup_freq, args.backup_freq)[::20]

    if env.d <= 2:
        means, stds = get_means_and_stds(env, data, iterations)

    # plot policy
    if env.d == 1:
        colors, labels = get_colors_and_labels(iterations, iter_str='Grad. iter.')
        plot_det_policies_1d(env, means, sol_hjb.u_opt, colors=colors, labels=labels, loc='upper left')

if __name__ == '__main__':
    main()
