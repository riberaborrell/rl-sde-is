from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.dpg_core import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # set action space bounds
    env.set_action_space_bounds()

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize state and action space (plot purposes only)
    env.discretize_state_space(h_state=0.05)
    env.discretize_action_space(h_action=0.05)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    control_hjb = np.expand_dims(sol_hjb.u_opt, axis=1)

    # run td3
    data = dpg(
        env=env,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_type=args.lr_type,
        n_iterations=args.n_iterations,
        seed=args.seed,
        test_freq_iterations=args.test_freq_iterations,
        test_batch_size=args.test_batch_size,
        backup_freq_iterations=args.backup_freq_iterations,
        value_function_opt= - sol_hjb.value_function,
        policy_opt=control_hjb,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plots
    if not args.plot:
        return

    # get models
    actor = data['actor']
    critic = data['critic']

    # get backup models
    if args.plot_episode is not None:
        load_backup_models(data, ep=args.plot_iteration)

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
                                        v_table_actor_critic, -sol_hjb.value_function)
    plot_advantage_function_1d(env, a_table, policy_critic)
    plot_det_policy_1d_actor_critic(env, policy_actor_init, policy_critic_init, policy_actor,
                                    policy_critic, sol_hjb.u_opt)

    # plot replay buffer
    plot_replay_buffer_1d(env, data['replay_states'][:, 0], data['replay_actions'][:, 0])

    # plot moving averages for each episode
    returns = data['returns']
    run_mean_returns = compute_running_mean(returns, 10)
    run_var_returns = compute_running_variance(returns, 10)
    time_steps = data['time_steps']
    run_mean_time_steps = compute_running_mean(time_steps, 10)
    plot_run_mean_returns_with_error_episodes(run_mean_returns, run_var_returns)
    plot_time_steps_episodes(time_steps, run_mean_time_steps)

    # plot expected values for each epoch
    test_mean_returns = data['test_mean_returns']
    test_var_returns = data['test_var_returns']
    test_mean_lengths = data['test_mean_lengths']
    test_policy_l2_errors = data['test_policy_l2_errors']
    plot_expected_returns_with_error_epochs(test_mean_returns, test_var_returns)
    plot_det_policy_l2_error_epochs(test_policy_l2_errors)


if __name__ == '__main__':
    main()
