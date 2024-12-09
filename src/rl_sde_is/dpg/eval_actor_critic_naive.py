import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy, evaluate_policy_torch
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.dpg.dpg_core import *

def main():
    args = get_base_parser().parse_args()

    # create gym envs 
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        alpha=args.alpha,
        beta=args.beta,
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatistics(env, args.test_batch_size, args.track_l2_error)

    # create object to store the is statistics of the learning
    is_stats = AISStatistics(args.test_freq, args.test_batch_size,
                             n_episodes=args.n_episodes, track_l2_error=args.track_l2_error)

    # load dpg with known q-value function
    data = dpg_actor_critic_naive(
        env,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=1.,
        seed=args.seed,
        n_episodes=args.n_episodes,
        load=True,
    )

    for i in range(is_stats.n_epochs):


        # load policy
        ep = i * is_stats.eval_freq
        load_backup_models(data, ep)

        # evaluate policy
        env.reset_statistics()
        evaluate_policy_torch(env, data['actor'], args.test_batch_size)

        # save and log epoch 
        is_stats.save_epoch(i, env)
        is_stats.log_epoch(i)
        env.close()

    # save is statistics
    is_stats.save_stats(data['dir_path'])


if __name__ == '__main__':
    main()
