import gymnasium as gym
import gym_sde_is
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl_sde_is.dpg.td3_core import *
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *


def main():
    parser = get_base_parser()
    parser.description = 'Run td3 for the sde importance sampling environment \
                          with a 1d toy example.'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        beta=args.beta,
        alpha=np.array(args.alpha),
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatistics(env, args.n_episodes)

    # discretize state and action space (plot purposes only)
    h_coarse = 0.1
    env.discretize_state_space(h_state=h_coarse)
    env.discretize_action_space(h_action=h_coarse)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    sol_hjb.coarse_solution(h_coarse)
    policy_opt = np.expand_dims(sol_hjb.u_opt, axis=1) if env.d == 1 else sol_hjb.u_opt
    value_function_opt = -sol_hjb.value_function

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
        value_function_opt=value_function_opt,
        policy_opt=policy_opt,
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

    # 1d-problems
    if env.d == 1:

        # get backup models
        if args.plot_episode is not None:
            load_backup_models(data, ep=args.plot_episode)

        # compute tables following q-value model
        q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic1)

        # compute value function and actions following the policy model
        v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic1)

        # load initial models
        load_backup_models(data, ep=0)

        # compute tables following q-value model
        _, v_table_critic_init, _, policy_critic_init = compute_tables_critic_1d(env, critic1)

        # compute value function and actions following the policy model
        v_table_actor_critic_init, policy_actor_init = compute_tables_actor_critic_1d(env, actor, critic1)


        plot_q_value_function_1d(env, q_table)
        plot_value_function_1d_actor_critic(env, v_table_critic_init, v_table_critic,
                                            v_table_actor_critic, value_function_opt)
        plot_advantage_function_1d(env, a_table, policy_critic)
        plot_det_policy_1d_actor_critic(env, policy_actor_init, policy_critic_init, policy_actor,
                                        policy_critic, policy_opt)

        # plot replay buffer
        plot_replay_buffer_1d(env, data['replay_states'][:, 0], data['replay_actions'][:, 0])


    elif env.d == 2:

        # compute actor policy
        states = torch.FloatTensor(env.state_space_h)
        policy = compute_det_policy_actions(env, actor, states)
        plot_det_policy_2d(env, policy, policy_opt)

        #TODO: fix this
        # compute tables
        #q_table, v_table, a_table, greed_actions = compute_tables_critic_2d(env, critic1)
        #plot_value_function_2d(env, v_table, value_function_opt)
        #plot_det_policy_2d(env, greed_actions, policy_opt)

        # plot states in replay buffer
        plot_replay_buffer_states_2d(env, data['replay_states'])


if __name__ == '__main__':
    main()
