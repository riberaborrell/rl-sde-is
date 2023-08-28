from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.td3_core import *

def get_policy(env, data, ep=None):
    actor = data['actor']
    if ep is not None:
        load_backup_models(data, ep)
        actor = data['actor']

    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    with torch.no_grad():
        policy = actor.forward(state_space_h).numpy().squeeze()
    return policy

def get_policies(env, data, episodes):

    Nx = env.n_states
    policies = np.empty((0, Nx), dtype=np.float32)

    for ep in episodes:
        load_backup_models(data, ep)
        policies = np.vstack((policies, get_policy(env, data).reshape(1, Nx)))

    return policies

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # set action space bounds
    env.set_action_space_bounds()

    # discretize state and action space (plot purposes only)
    env.discretize_state_space(h_state=0.1)
    env.discretize_action_space(h_action=0.1)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    policy_opt = np.expand_dims(sol_hjb.u_opt, axis=1)
    value_function_opt = -sol_hjb.value_function

    # run td3
    data = td3_episodic(
        env=env,
        gamma=args.gamma,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        start_steps=int(1e3),
        replay_size=args.replay_size,
        update_after=int(1e3),
        n_steps_episode_lim=args.n_steps_lim,
        update_every=10,
        expl_noise_init=args.expl_noise_init,
        expl_noise_decay=args.decay,
        policy_delay=args.policy_delay,
        target_noise=args.target_noise,
        action_limit=args.action_limit,
        polyak=args.polyak,
        test_freq_episodes=args.test_freq_episodes,
        test_batch_size=args.test_batch_size,
        backup_freq_episodes=args.backup_freq_episodes,
        value_function_opt=value_function_opt,
        policy_opt=policy_opt,
        load=args.load,
        test=args.test,
        live_plot=args.live_plot,
    )

    # plots
    if not args.plot:
        return


    # get models
    actor = data['actor']
    critic1 = data['critic1']
    critic2 = data['critic2']

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


if __name__ == '__main__':
    main()
