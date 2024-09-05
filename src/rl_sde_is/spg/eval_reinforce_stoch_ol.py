import gymnasium as gym
import numpy as np

import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_torch_vect, evaluate_gaussian_policy_torch_vect
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.spg.reinforce_stochastic_core import reinforce_stochastic, load_backup_model

def main():
    parser = get_base_parser()
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        alpha=args.alpha,
        beta=args.beta,
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatisticsVect(env, args.eval_batch_size, args.track_l2_error)

    # create object to store the is statistics of the evaluation
    is_stats = ISStatistics(
        args.eval_freq, args.eval_batch_size, args.n_grad_iterations,
        policy_type=args.policy_type, iter_str='grad. it.:', track_l2_error=args.track_l2_error,
    )

    # load reinforce with gaussian stochastic policy
    data = reinforce_stochastic(
        env,
        expectation_type=args.expectation_type,
        return_type=args.return_type,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        theta_init=args.theta_init,
        policy_type=args.gaussian_policy_type,
        policy_noise=args.policy_noise,
        estimate_z=args.estimate_z,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        n_grad_iterations=args.n_grad_iterations,
        learn_value=args.learn_value,
        seed=args.seed,
        load=True,
    )

    # evaluate policy by fixing the initial position
    if args.state_init_dist == 'uniform':
        env.unwrapped.state_init_dist = 'delta'

    for i in range(is_stats.n_epochs):

        # load policy
        succ = load_backup_model(data, i * is_stats.eval_freq)

        # break if the model was not loaded
        if not succ:
            break

        # evaluate policy
        if args.policy_type == 'stoch':
            evaluate_gaussian_policy_torch_vect(env, data['policy'], args.eval_batch_size)
        else:
            evaluate_policy_torch_vect(env, data['policy'].mean, args.eval_batch_size)

        # save and log epoch 
        is_stats.save_epoch(i, env)
        is_stats.log_epoch(i)
        env.close()

    # save is statistics
    is_stats.save_stats(data['dir_path'])


if __name__ == '__main__':
    main()
