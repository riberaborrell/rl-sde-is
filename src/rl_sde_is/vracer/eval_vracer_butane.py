import gymnasium as gym
import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics
import numpy as np

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import AISStatistics
from rl_sde_is.vracer.vracer_utils import *
from rl_sde_is.vracer.load_model import load_model

def main():

    args = get_base_parser().parse_args()

    # create object to store the is statistics of the learning
    ais_stats = AISStatistics(args.n_episodes, args.test_freq_episodes, args.test_batch_size)

    for i in range(ais_stats.n_epochs):

        # create gym envs 
        env = gym.make(
            'sde-is-butane-mgf-v0',
            temperature=args.temperature,
            gamma=10.0,
        )
        env = RecordEpisodeStatistics(env, args.test_batch_size)

        # load policy
        ep = i * ais_stats.eval_freq_episodes
        results_dir = get_vracer_rel_dir_path(env, args)
        model = load_model(results_dir + '/gen{}.json'.format(str(ep).zfill(8)))

        # evaluate policy
        evaluate_policy(env, model, args.test_batch_size)

        # save and log epoch 
        ais_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                             np.exp(env.log_psi_is))
        ais_stats.log_epoch(i)
        env.close()

    # save is statistics
    dir_path = get_vracer_dir_path(env, args)
    ais_stats.save_stats(dir_path)


if __name__ == '__main__':
    main()
