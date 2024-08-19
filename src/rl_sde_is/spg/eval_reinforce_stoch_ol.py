import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_torch_vect, evaluate_gaussian_policy_torch_vect
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.spg.spg_core import *

def main():
    parser = get_base_parser()
    parser.add_argument(
        '--algorithm-type',
        choices=['initial-return', 'n-return'],
        default='initial-return',
        help='Set reinforce stoch algorithm type. Default: initial-return',
    )
    parser.add_argument(
        '--expectation-type',
        choices=['random-time', 'on-policy'],
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
    env = RecordEpisodeStatisticsVect(env, args.eval_batch_size, args.track_l2_error)

    # create object to store the is statistics of the evaluation
    is_stats = ISStatistics(
        eval_freq=args.eval_freq,
        eval_batch_size=args.eval_batch_size,
        policy_type=args.policy_type,
        n_grad_iterations=args.n_grad_iterations,
        track_l2_error=args.track_l2_error,
    )

    # load reinforce with gaussian stochastic policy
    data = reinforce_stochastic(
        env,
        algorithm_type=args.algorithm_type,
        expectation_type=args.expectation_type,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        theta_init=args.theta_init,
        policy_type=args.gaussian_policy_type,
        policy_noise=args.policy_noise,
        estimate_mfht=args.estimate_mfht,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        seed=args.seed,
        n_grad_iterations=args.n_grad_iterations,
        load=True,
    )

    # evaluate policy by fixing the initial position
    if args.state_init_dist == 'uniform':
        env.unwrapped.state_init_dist = 'delta'

    for i in range(is_stats.n_epochs):

        # load policy
        j = i * is_stats.eval_freq
        load_backup_model(data, j)

        # evaluate policy
        if args.policy_type == 'stoch':
            evaluate_gaussian_policy_torch_vect(env, data['policy'], args.eval_batch_size)
        else:
            evaluate_policy_torch_vect(env, data['policy'].mean, args.eval_batch_size)

        # save and log epoch 
        is_stats.save_epoch(i, env)
        is_stats.log_epoch(i)
        env.close()

    # save is statistics
    is_stats.save_stats(data['dir_path'])


if __name__ == '__main__':
    main()
