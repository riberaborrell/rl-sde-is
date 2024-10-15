import gymnasium as gym
import gym_sde_is

from rl_sde_is.dpg.td3_core import td3_episodic
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.description = 'Run td3 for the sde importance sampling environment \
                          with a ol toy example (without hjb reference solution).'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        d=args.d,
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        state_init_dist=args.state_init_dist,
    )

    # run td3
    data = td3_episodic(
        env=env,
        gamma=args.gamma,
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
    if not args.plot:
        return

    # plot returns and time steps
    x = np.arange(data['n_episodes'])
    plot_y_per_episode(x, data['returns'], run_window=10, title='Returns', legend=True)
    plot_y_per_episode(x, data['time_steps'], run_window=10, title='Time steps')

if __name__ == '__main__':
    main()
