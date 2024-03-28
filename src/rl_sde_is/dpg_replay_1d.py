from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.dpg_core import dpg_replay

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

    # run simple dpg with known q-value function
    data = dpg_replay(
        env=env,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        lr_actor=args.lr_actor,
        n_steps_episode_lim=args.n_steps_lim,
        n_episodes=args.n_episodes,
        seed=args.seed,
        test_freq_episodes=args.test_freq_episodes,
        test_batch_size=args.test_batch_size,
        backup_freq_episodes=args.backup_freq_episodes,
        value_function_opt= - sol_hjb.value_function,
        policy_opt=control_hjb,
        load=args.load,
        test=args.test,
        live_plot=args.live_plot,
    )

    # plots
    if not args.plot:
        return

    # get models
    actor = data['actor']

    # plot expected values for each epoch
    test_mean_returns = data['test_mean_returns']
    test_var_returns = data['test_var_returns']
    test_policy_l2_errors = data['test_policy_l2_errors']
    plot_expected_returns_with_error_epochs(test_mean_returns, test_var_returns)
    plot_det_policy_l2_error_epochs(test_policy_l2_errors)


if __name__ == '__main__':
    main()
