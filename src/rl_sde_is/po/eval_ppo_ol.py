import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_torch_vect, evaluate_gaussian_policy_torch_vect
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

from rl_sde_is.po.ppo_core import PPO
from rl_sde_is.po.ppo_parser import add_ppo_arguments
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics

def main():
    parser = get_base_parser()
    add_ppo_arguments(parser)
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        T=args.T,
        reward_type=args.reward_type,
        state_init_dist=args.state_init_dist,
        n_steps_lim=args.n_steps_lim,
    )
    env = RecordEpisodeStatisticsVect(env, args.eval_batch_size, args.track_l2_error)

    # load ppo
    agent = PPO(
        env,
        gamma=args.gamma,
        policy_noise_init=args.policy_noise,
        n_layers=args.n_layers,
        #d_hidden_layer=args.d_hidden_layer,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        update_epochs=args.update_epochs,
        n_mini_batches=args.n_mini_batches,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        optim_type=args.optim_type,
        norm_adv=args.norm_adv,
        clip_vloss=args.clip_vloss,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        seed=args.seed,
    )
    succ, data = agent.run_ppo(load=True)
    if not succ:
        return

    # create object to store the is statistics of the learning
    assert args.policy_type in ['stoch', 'stoch-mean'], 'Policy type not recognized'
    is_stats = ISStatistics(args.eval_freq, args.eval_batch_size, args.n_iterations, iter_str='it',
                            policy_type=args.policy_type, track_l2_error=args.track_l2_error)

    # evaluate policy by fixing the initial position
    if args.state_init_dist == 'uniform':
        env.unwrapped.state_init_dist = 'delta'

    for i in range(is_stats.n_epochs):

        # load policy
        succ = agent.load_backup_model(data, i * is_stats.eval_freq)

        # break if the model was not loaded
        if not succ:
            break

        # evaluate policy
        if args.policy_type == 'stoch':
            evaluate_gaussian_policy_torch_vect(env, agent.model.actor, args.eval_batch_size)
        else:
            evaluate_policy_torch_vect(env, agent.model.actor.mean, args.eval_batch_size)

        # save and log epoch 
        is_stats.save_epoch(i, env)
        is_stats.log_epoch(i)

    # save is statistics
    is_stats.save_eval_stats(data['dir_path'])

    # close env
    env.close()

if __name__ == '__main__':
    main()
