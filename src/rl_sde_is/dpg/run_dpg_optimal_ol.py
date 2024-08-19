import gymnasium as gym
import gym_sde_is

from rl_sde_is.dpg.dpg_core import dpg_optimal
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.add_argument(
        '--expectation-type',
        choices=['random-time', 'on-policy'],
        default='random-time',
        help='Set type of expectation. Default: random-time',
    )
    parser.add_argument(
        '--mini-batch-size',
        type=int,
        default=None,
        help='Set mini batch size for on-policy expectations. Default: None',
    )
    parser.add_argument(
        '--estimate-mfht',
        action='store_true',
        help='Estimate the mfht in the dpg.',
    )
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
    h_coarse = 0.05
    env.discretize_state_space(h_state=0.05)
    env.discretize_action_space(h_action=0.05)

    # get hjb solver
    sol_hjb = env.get_hjb_solver(h_coarse)

    # run dpg with known q-value function
    data = dpg_optimal(
        env,
        expectation_type=args.expectation_type,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        estimate_mfht=args.estimate_mfht,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        seed=args.seed,
        n_grad_iterations=args.n_grad_iterations,
        backup_freq=args.backup_freq,
        value_function_opt= - sol_hjb.value_function,
        policy_opt=sol_hjb.u_opt,
        live_plot_freq=args.live_plot_freq,
        load=args.load,
    )

    # plots
    if not args.plot:
        return

    # plot statistics
    x = np.arange(data['n_grad_iterations']+1)
    plot_y_per_grad_iteration(x, data['mean_returns'], title='Objective function')
    plot_y_per_grad_iteration(x, data['mean_fhts'], title='MFHT')

if __name__ == '__main__':
    main()
