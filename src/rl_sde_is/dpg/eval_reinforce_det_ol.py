import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_torch_vect
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.dpg.reinforce_deterministic_core import *

def main():
    args = get_base_parser().parse_args()

    # create gym envs 
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
        n_grad_iterations=args.n_grad_iterations,
        track_l2_error=args.track_l2_error,
    )

    # load reinforce algorithm with a deterministic policy
    data = reinforce_deterministic(
        env,
        expectation_type=args.expectation_type,
        return_type=args.return_type,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        n_grad_iterations=args.n_grad_iterations,
        seed=args.seed,
        learn_value=args.learn_value,
        mini_batch_size=args.mini_batch_size,
        memory_size=args.replay_size,
        estimate_mfht=args.estimate_mfht,
        lr_value=args.lr_value,
        load=True,
    )

    # evaluate policy by fixing the initial position
    if args.state_init_dist == 'uniform':
        env.unwrapped.state_init_dist = 'delta'

    for i in range(is_stats.n_epochs):

        # load policy
        load_backup_model(data, i * is_stats.eval_freq)

        # evaluate policy
        evaluate_policy_torch_vect(env, data['model'], args.eval_batch_size)

        # save and log epoch 
        is_stats.save_epoch(i, env)
        is_stats.log_epoch(i)

    # save is statistics
    is_stats.save_stats(data['dir_path'])

    # close env
    env.close()


if __name__ == '__main__':
    main()
