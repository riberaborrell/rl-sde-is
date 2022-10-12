from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.ddpg_core import *
from rl_sde_is.environments_2d import DoubleWellStoppingTime2D
from rl_sde_is.plots import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime2D(alpha=args.alpha, beta=args.beta)

    # set action space bounds
    env.action_space_low = -5
    env.action_space_high = 5

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize state and action space (plot purposes only)
    env.discretize_state_space(h_state=0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run ddpg 
    data = ddpg_episodic(
        env=env,
        gamma=args.gamma,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        start_steps=10000,
        replay_size=100000,
        update_after=10000,
        update_every=100,
        n_steps_episode_lim=1000,
        test_freq_episodes=100,
        test_batch_size=1000,
        backup_freq_episodes=args.backup_freq_episodes,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )

    # plots
    if not args.plot:
        return

    # get models
    actor = data['actor']
    critic = data['critic']

    # get backup models
    if args.plot_episode is not None:
        load_backup_models(data, ep=args.plot_episode)


    # compute actor policy
    states = torch.FloatTensor(env.state_space_h)
    policy = compute_det_policy_actions(env, actor, states)
    plot_det_policy_2d(env, policy)

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
