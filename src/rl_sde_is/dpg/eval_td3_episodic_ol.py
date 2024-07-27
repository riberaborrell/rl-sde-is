import gymnasium as gym
import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_torch_vect
from gym_sde_is.utils.sde import compute_is_functional
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
import numpy as np

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.dpg.td3_core import *

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

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(args.eval_freq, args.eval_batch_size,
                            n_episodes=args.n_episodes, track_l2_error=args.track_l2_error)

    # load td3
    data = td3_episodic(
        env,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        n_steps_lim=args.n_steps_lim,
        policy_freq=args.policy_freq,
        target_noise=args.target_noise,
        action_limit=args.action_limit,
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
        ep = i * is_stats.eval_freq
        load_backup_models(data, ep)
        policy = data['actor']

        # evaluate policy
        evaluate_policy_torch_vect(env, policy, args.eval_batch_size)

        # save and log epoch 
        l2_errors = env.l2_errors if args.track_l2_error else None
        is_functional = compute_is_functional(env.girs_stoch_int,
                                              env.running_rewards, env.terminal_rewards)
        is_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                            is_functional=is_functional, l2_errors=l2_errors)
        is_stats.log_epoch(i)

    # save is statistics
    is_stats.save_stats(data['dir_path'])

    # close env
    env.close()

if __name__ == '__main__':
    main()
