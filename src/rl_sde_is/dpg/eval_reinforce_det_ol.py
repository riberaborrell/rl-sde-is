import gymnasium as gym
import gym_sde_is
from gym_sde_is.utils.evaluate import evaluate_policy_torch_vect
from gym_sde_is.utils.sde import compute_is_functional
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
import numpy as np

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.dpg.reinforce_deterministic_core import *

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

    # create object to store the is statistics of the evaluation
    is_stats = ISStatistics(
        eval_freq=args.eval_freq,
        eval_batch_size=args.eval_batch_size,
        n_grad_iterations=args.n_grad_iterations,
        track_l2_error=args.track_l2_error,
    )

    # load reinforve algorithm with a deterministic policy
    data = reinforce_deterministic(
        env,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        n_grad_iterations=args.n_grad_iterations,
        seed=args.seed,
        load=True,
    )

    for i in range(is_stats.n_epochs):

        # load policy
        j = i * is_stats.eval_freq
        load_backup_model(data, j)

        # evaluate policy
        evaluate_policy_torch_vect(env, data['model'], args.eval_batch_size)

        # save and log epoch 
        l2_errors = env.l2_errors if args.track_l2_error else None
        is_functional = compute_is_functional(env.girs_stoch_int,
                                              env.running_rewards, env.terminal_rewards)
        is_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                            is_functional=is_functional, l2_errors=l2_errors)
        is_stats.log_epoch(i)
        env.close()

    # save is statistics
    is_stats.save_stats(data['dir_path'])


if __name__ == '__main__':
    main()
