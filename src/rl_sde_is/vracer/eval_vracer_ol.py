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
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        T=args.T,
        reward_type=args.reward_type,
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatisticsVect(env, args.eval_batch_size, args.track_l2_error)

    # load vracer
    data = vracer(env, args, load=True)

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(args.eval_freq, args.eval_batch_size,
                            n_episodes=args.n_episodes, track_l2_error=args.track_l2_error)

    # evaluate policy by fixing the initial position
    env.state_init_dist = 'delta'
    for i in range(is_stats.n_epochs):

        # load policy
        ep = i * is_stats.eval_freq
        model = load_model(args.rel_dir_path + '/model{}.json'.format(str(ep).zfill(8)))

        # evaluate policy
        evaluate_policy_vect(env, model.mean, args.eval_batch_size)

        # save and log epoch 
        l2_errors = env.l2_errors if args.track_l2_error else None
        is_functional = compute_is_functional(env.girs_stoch_int,
                                              env.running_rewards, env.terminal_rewards)
        is_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                            is_functional=is_functional, l2_errors=l2_errors)
        is_stats.log_epoch(i)

    # save is statistics
    is_stats.save_stats(args.dir_path)

    # close env
    env.close()

if __name__ == '__main__':
    main()
