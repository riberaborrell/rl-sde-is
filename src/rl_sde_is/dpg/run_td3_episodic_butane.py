import gymnasium as gym
import gym_sde_is

from rl_sde_is.dpg.td3_core import td3_episodic
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.description = 'Run td3 for the sde importance sampling environment \
                          with the butane molecule.'
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

    # run td3
    succ, data = td3_episodic(
        env=env,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        replay_size=args.replay_size,
        learning_starts=args.learning_starts,
        n_steps_lim=args.n_steps_lim,
        update_freq=args.update_freq,
        expl_noise_init=args.expl_noise_init,
        expl_noise_decay=args.decay,
        policy_freq=args.policy_freq,
        target_noise=args.target_noise,
        action_limit=args.action_limit,
        polyak=args.polyak,
        backup_freq=args.backup_freq,
        log_freq=args.log_freq,
        load=args.load,
    )

    # plots
    if not args.plot or not succ:
        return

    # plot returns and time steps
    x = np.arange(data['n_episodes'])
    plot_y_per_episode(x, data['returns'], run_window=10, title='Returns', legend=True)
    plot_y_per_episode(x, data['time_steps'], run_window=10, title='Time steps')

if __name__ == '__main__':
    main()
