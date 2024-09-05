import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_torch_vect
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

from rl_sde_is.dpg.ddpg_core import ddpg_episodic, load_backup_models
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics

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
    is_stats = ISStatistics(args.eval_freq, args.eval_batch_size, args.n_episodes,
                            iter_str='ep.:', track_l2_error=args.track_l2_error)

    # load ddpg
    data = ddpg_episodic(
        env,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        n_steps_lim=args.n_steps_lim,
        expl_noise=args.expl_noise_init,
        polyak=args.polyak,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        load=True,
    )

    # evaluate policy by fixing the initial position
    if args.state_init_dist == 'uniform':
        env.unwrapped.state_init_dist = 'delta'

    for i in range(is_stats.n_epochs):

        # load policy
        succ = load_backup_models(data, i * is_stats.eval_freq)

        # break if the model was not loaded
        if not succ:
            break

        # evaluate policy
        evaluate_policy_torch_vect(env, data['actor'], args.eval_batch_size)

        # save and log epoch 
        is_stats.save_epoch(i, env)
        is_stats.log_epoch(i)

    # save is statistics
    is_stats.save_stats(data['dir_path'])

    # close env
    env.close()

if __name__ == '__main__':
    main()
