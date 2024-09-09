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
        d=args.d,
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        state_init_dist=args.state_init_dist,
    )

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
        estimate_z=args.estimate_z,
        lr_value=args.lr_value,
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        live_plot_freq=args.live_plot_freq,
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


if __name__ == "__main__":
    main()
