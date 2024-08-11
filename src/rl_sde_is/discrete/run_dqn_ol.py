import gymnasium as gym

import gym_sde_is

from rl_sde_is.discrete.dqn_core import dqn
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.description = 'Run dqn for the sde importance sampling environment \
                          with a ol toy example.'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        state_init_dist=args.state_init_dist,
    )

    # discretize action space
    env.discretize_action_space(args.h_action)

    # run dqn 
    data = dqn(
        env=env,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        n_episodes=args.n_episodes,
        replay_size=args.replay_size,
        learning_starts=args.learning_starts,
        n_steps_lim=args.n_steps_lim,
        update_freq=args.update_freq,
        expl_noise=args.expl_noise_init,
        action_limit=args.action_limit,
        polyak=args.polyak,
        seed=args.seed,
        load=args.load
    )

    n_episodes = data['n_episodes']
    returns = data['returns']
    time_steps = data['time_steps']

    x = np.arange(n_episodes)

    # plot returns and time steps
    plot_y_per_episode_with_run_mean(x, returns, run_window=10, title='Returns', legend=True)
    plot_y_per_episode_with_run_mean(x, time_steps, run_window=10, title='Time steps')
    return

    # compute value function and greedy actions
    obs = torch.arange(-2, 2+0.01, 0.01).unsqueeze(dim=1)
    phi = model.forward(obs).detach()
    value_function = torch.max(phi, axis=1)[0].numpy()
    policy = torch.argmax(phi, axis=1).numpy()
    actions = env.action_space_h[policy]

    # plot v function
    x = obs.squeeze().numpy()
    plt.plot(x, value_function)
    plt.show()

    # plot control
    plt.plot(x, actions)
    plt.show()


if __name__ == '__main__':
    main()
