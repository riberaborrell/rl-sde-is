import gymnasium as gym
import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy, evaluate_policy_torch
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym_sde_is.utils.sde import compute_is_functional
import numpy as np

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.dpg.td3_core import *

def main():
    args = get_base_parser().parse_args()

    # create gym envs 
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        alpha=np.array(args.alpha),
        beta=args.beta,
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatistics(env, args.test_batch_size, args.track_l2_error)

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(args.n_episodes, args.test_freq, args.test_batch_size,
                              args.track_l2_error)

    # load td3
    data = td3_episodic(
        env,
        d_hidden_layer=args.d_hidden,
        n_steps_lim=args.n_steps_lim,
        policy_freq=args.policy_freq,
        target_noise=args.target_noise,
        expl_noise_init=args.expl_noise_init,
        polyak=args.polyak,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        load=True,
    )

    for i in range(is_stats.n_epochs):


        # load policy
        ep = i * is_stats.eval_freq_episodes
        load_backup_models(data, ep)
        policy = data['actor']

        # evaluate policy
        env.reset_statistics()
        evaluate_policy_torch(env, policy, args.test_batch_size)

        # save and log epoch 
        l2_errors = env.l2_errors if args.track_l2_error else None
        is_functional = compute_is_functional(env.girs_stoch_int,
                                              env.running_rewards, env.terminal_rewards)
        is_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                             is_functional, l2_errors)
        is_stats.log_epoch(i)
        env.close()

    # save is statistics
    is_stats.save_stats(data['rel_dir_path'])


if __name__ == '__main__':
    main()
