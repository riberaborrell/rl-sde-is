import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_vect
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.vracer.vracer_utils import *
from rl_sde_is.vracer.load_model import load_model

def main():

    args = get_base_parser().parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-butane-{}-v0'.format(args.setting),
        temperature=args.temperature,
        gamma=10.0,
        T=args.T,
        state_init_dist=args.state_init_dist,
        n_steps_lim=args.n_steps_lim,
    )
    env = RecordEpisodeStatisticsVect(env, args.eval_batch_size)

    # load vracer
    data = vracer(env, args, load=True)

    # create object to store the is statistics of the learning
    assert args.policy_type in ['stoch', 'stoch-mean'], 'Policy type not recognized'
    is_stats = ISStatistics(args.eval_freq, args.eval_batch_size, args.n_episodes,
                            iter_str='ep', policy_type=args.policy_type)

    # evaluate policy by fixing the initial position
    if args.state_init_dist == 'uniform':
        env.unwrapped.state_init_dist = 'delta'

    for i in range(is_stats.n_epochs):

        # load policy
        succ, model = load_model(rel_dir_path=args.rel_dir_path, ep=i * is_stats.eval_freq)

        # break if the model was not loaded
        if not succ:
            break

        # evaluate policy
        if args.policy_type == 'stoch':
            evaluate_gaussian_policy_vect(env, model, args.eval_batch_size)
        else:
            evaluate_policy_vect(env, model.mean, args.eval_batch_size)

        # save and log epoch 
        is_stats.save_epoch(i, env)
        is_stats.log_epoch(i)

    # save is statistics
    is_stats.save_stats(args.dir_path)

    # close env
    env.close()

if __name__ == '__main__':
    main()
