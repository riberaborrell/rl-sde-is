import gymnasium as gym
import numpy as np

import gym_sde_is

from rl_sde_is.po.ppo_core import PPO
from rl_sde_is.po.ppo_parser import add_ppo_arguments
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.description = 'Run ppo for the sde importance sampling environment \
                          with the butane molecule'
    add_ppo_arguments(parser)
    args = parser.parse_args()

    assert args.dt <= 5e-4

    # create gym environment
    env = gym.make(
        'sde-is-butane-{}-v0'.format(args.setting),
        dt=args.dt,
        is_reduced=args.is_reduced,
        temperature=args.temperature,
        gamma=10.0,
        T=args.T,
        state_init_dist=args.state_init_dist,
    )

    # PPO agent
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
        cuda=args.cuda,
        torch_deterministic=args.torch_deterministic,
    )

    # run
    succ, data = agent.run_ppo(
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # plot statistics
    x = np.arange(data['n_iterations'] + 1)
    plot_y_per_x(x, data['mean_returns'], title='Returns', run_window=1, legend=True)
    plot_y_per_x(x, data['mean_fhts'], title='MFHT')


if __name__ == "__main__":
    main()
