import gymnasium as gym
import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_vect
from gym_sde_is.utils.sde import compute_is_functional
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
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
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatisticsVect(env, args.eval_batch_size)

    # load vracer
    data = vracer(env, args, load=True)

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(args.eval_freq, args.eval_batch_size, n_episodes=args.n_episodes)

    # evaluate policy by fixing the initial position
    if args.state_init_dist == 'uniform':
        env.unwrapped.state_init_dist = 'delta'

    for i in range(is_stats.n_epochs):

        # load policy
        ep = i * is_stats.eval_freq
        model = load_model(args.rel_dir_path + '/model{}.json'.format(str(ep).zfill(8)))

        # evaluate policy
        evaluate_policy_vect(env, model.mean, args.eval_batch_size)

        # save and log epoch 
        is_functional = compute_is_functional(
            env.girs_stoch_int, env.running_rewards, env.terminal_rewards,
        )
        is_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                            is_functional=is_functional)
        is_stats.log_epoch(i)

    # save is statistics
    is_stats.save_stats(args.dir_path)

    # close env
    env.close()

if __name__ == '__main__':
    main()
