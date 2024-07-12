import gymnasium as gym
import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy
from gym_sde_is.utils.sde import compute_is_functional
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics
import numpy as np

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.vracer.vracer_utils import *
from rl_sde_is.vracer.load_model import load_model

def main():

    args = get_base_parser().parse_args()

    # create gym envs 
    env = gym.make(
        'sde-is-butane-{}-v0'.format(args.setting),
        temperature=args.temperature,
        gamma=10.0,
        T=args.T,
    )
    env = RecordEpisodeStatistics(env, args.test_batch_size)

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(args.test_freq, args.test_batch_size, n_episdoes=args.n_episodes)

    for i in range(is_stats.n_epochs):


        # load policy
        ep = i * is_stats.eval_freq
        results_dir = get_vracer_rel_dir_path(env, args)
        model = load_model(results_dir + '/gen{}.json'.format(str(ep).zfill(8)))

        # evaluate policy
        env.reset_statistics()
        evaluate_policy(env, model.policy, args.test_batch_size)

        # save and log epoch 
        is_functional = compute_is_functional(
            env.girs_stoch_int, env.running_rewards, env.terminal_rewards,
        )
        is_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                             is_functional)
        is_stats.log_epoch(i)
        env.close()

    # save is statistics
    dir_path = get_vracer_dir_path(env, args)
    is_stats.save_stats(dir_path)


if __name__ == '__main__':
    main()
