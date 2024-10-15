import gymnasium as gym
import gym_sde_is

from rl_sde_is.dpg.ddpg_core import *
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *


def main():
    parser = get_base_parser()
    parser.description = 'Run ddpg for the sde importance sampling environment \
                          with a ol toy example (with hjb reference solution).'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        state_init_dist=args.state_init_dist,
    )

    # discretize state and action space (plot purposes only)
    h_coarse = 0.1
    env.discretize_state_space(h_state=h_coarse)
    env.discretize_action_space(h_action=h_coarse)

    # get hjb solver
    sol_hjb = env.get_hjb_solver(h_coarse)

    # run ddpg
    data = ddpg_episodic(
        env=env,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        n_steps_lim=args.n_steps_lim,
        seed=args.seed,
        replay_size=args.replay_size,
        learning_starts=args.learning_starts,
        update_freq=args.update_freq,
        expl_noise=args.expl_noise_init,
        action_limit=args.action_limit,
        polyak=args.polyak,
        backup_freq=args.backup_freq,
        log_freq=args.log_freq,
        live_plot_freq=args.live_plot_freq,
        value_function_opt=-sol_hjb.value_function,
        policy_opt=sol_hjb.u_opt,
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
