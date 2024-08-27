import gymnasium as gym
import gym_sde_is

from rl_sde_is.dpg.reinforce_deterministic_core import reinforce_deterministic, \
                                                       get_policies, get_value_functions
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *


def main():
    parser = get_base_parser()
    parser.description = 'Run model-based reinforce for deterministic policies for the sde \
                          importance sampling environment with a ol toy example.'
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
    h_coarse = 0.01
    env.discretize_state_space(h_state=h_coarse)
    env.discretize_action_space(h_action=h_coarse)

    # get hjb solver
    sol_hjb = env.get_hjb_solver(h_coarse)

    # run reinforce algorithm with a deterministic policy
    data = reinforce_deterministic(
        env,
        expectation_type=args.expectation_type,
        return_type=args.return_type,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        theta_init=args.theta_init,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        memory_size=args.replay_size,
        lr=args.lr,
        n_grad_iterations=args.n_grad_iterations,
        seed=args.seed,
        learn_value=args.learn_value,
        estimate_mfht=args.estimate_mfht,
        lr_value=args.lr_value,
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        live_plot_freq=args.live_plot_freq,
        policy_opt=sol_hjb.u_opt,
        value_function_opt=-sol_hjb.value_function,
        load=args.load,
    )

    # do plots
    if not args.plot:
        return

    # get backup iterations
    iterations = np.arange(0, args.n_grad_iterations + args.backup_freq, args.backup_freq)

    # plot statistics
    x = np.arange(data['n_grad_iterations']+1)
    plot_y_per_grad_iteration(x, data['mean_returns'], title='Objective function')
    plot_y_per_grad_iteration(x, data['var_returns'], title='Variance of initial return')
    plot_y_per_grad_iteration(x, data['losses'], title='Effective loss')
    plot_y_per_grad_iteration(x, data['loss_vars'], title='Effective loss (variance)')
    plot_y_per_grad_iteration(x, data['mean_fhts'], title='MFHT')
    plot_y_per_grad_iteration(x, data['re_I_us'], title=r'Sampled relative error $\widehat{Re}$')

    # plot policy
    if env.d <= 2:
        policies = get_policies(env, data, iterations)
        if args.learn_value:
            value_functions = get_value_functions(env, data, iterations)

    if env.d == 1:
        plot_det_policies_1d(env, policies, sol_hjb.u_opt)
        if args.learn_value:
            plot_ys_1d(env, value_functions, -sol_hjb.value_function)

    if env.d == 2:
        plot_det_policy_2d(env, policies[0].reshape(env.n_states_axis+(env.d,)), sol_hjb.u_opt)
        plot_det_policy_2d(env, policies[-1].reshape(env.n_states_axis+(env.d,)), sol_hjb.u_opt)

if __name__ == "__main__":
    main()
