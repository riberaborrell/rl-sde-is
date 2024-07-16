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
        alpha=np.array(args.alpha),
        beta=args.beta,
        T=args.T,
        reward_type=args.reward_type,
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatisticsVect(env, args.test_batch_size, args.track_l2_error)

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(args.test_freq, args.test_batch_size,
                            n_episodes=args.n_episodes, track_l2_error=args.track_l2_error)

    for i in range(is_stats.n_epochs):


        # load policy
        ep = i * is_stats.eval_freq
        results_dir = get_vracer_rel_dir_path(env, args)
        model = load_model(results_dir + '/model{}.json'.format(str(ep).zfill(8)))

        # evaluate policy
        evaluate_policy_vect(env, model.policy, args.test_batch_size)

        # save and log epoch 
        l2_errors = env.l2_errors if args.track_l2_error else None
        is_functional = compute_is_functional(env.girs_stoch_int,
                                              env.running_rewards, env.terminal_rewards)
        is_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                            is_functional=is_functional, l2_errors=l2_errors)
        is_stats.log_epoch(i)
        env.close()

    # save is statistics
    dir_path = get_vracer_dir_path(env, args)
    is_stats.save_stats(dir_path)


if __name__ == '__main__':
    main()
