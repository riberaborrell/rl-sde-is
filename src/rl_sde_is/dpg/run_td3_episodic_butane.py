import gymnasium as gym
import gym_sde_is
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl_sde_is.dpg.td3_core import *
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *


def main():
    parser = get_base_parser()
    parser.description = 'Run td3 for the sde importance sampling environment \
                          with the butane molecule.'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-butane-{}-v0'.format(args.setting),
        temperature=args.temperature,
        gamma=10.0,
        T=args.T,
    )

    # run td3
    data = td3_episodic(
        env=env,
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
        load=args.load,
        live_plot=args.live_plot,
    )

    # plots
    if not args.plot:
        return

    # plot moving averages for each episode
    n_episodes = data['n_episodes']
    returns = data['returns']
    time_steps = data['time_steps']

    x = np.arange(n_episodes)

    # plot returns and time steps
    plot_y_per_episode_with_run_mean(x, returns, run_window=10, title='Returns', legend=True)
    plot_y_per_episode_with_run_mean(x, time_steps, run_window=10, title='Time steps')

    # get models
    actor = data['actor']
    critic1 = data['critic1']
    critic2 = data['critic2']

if __name__ == '__main__':
    main()
