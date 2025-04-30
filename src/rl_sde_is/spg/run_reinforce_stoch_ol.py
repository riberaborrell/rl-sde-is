import gymnasium as gym
import gym_sde_is

import numpy as np

from rl_sde_is.spg.reinforce_stochastic_core import ReinforceStochastic
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.description = 'Run reinforce stochastic for the sde importance sampling environment'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        alpha=args.alpha,
        beta=args.beta,
        state_init_dist=args.state_init_dist,
    )

    # discretize state and action space (plot purposes only)
    env.unwrapped.discretize_state_space(h_state=args.h_state)

    # get hjb solver
    sol_hjb = env.unwrapped.get_hjb_solver(args.h_state)

    # run reinforce with gaussian stochastic policy
    agent = ReinforceStochastic(
        env,
        gamma=args.gamma,
        expectation_type=args.expectation_type,
        return_type=args.return_type,
        estimate_z=args.estimate_z,
        policy_type=args.gaussian_policy_type,
        policy_noise=args.policy_noise,
        theta_init=args.theta_init,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        optim_type=args.optim_type,
        batch_size=args.batch_size,
        mini_batch_size_type=args.mini_batch_size_type,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        learn_value=args.learn_value,
        lr_value=args.lr_value,
        n_grad_iterations=args.n_grad_iterations,
        seed=args.seed,
    )
    succ, data = agent.run_agent(
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        live_plot_freq=args.live_plot_freq,
        policy_opt=sol_hjb.u_opt,
        load=args.load,
    )

    # plot results
    if not args.plot or not succ:
        return

    # plot statistics
    x = np.arange(data['n_grad_iterations']+1)
    plot_y_per_grad_iteration(x, data['mean_returns'], title='Objective function')
    plot_y_per_grad_iteration(x, data['losses'], title='Effective loss')
    plot_y_per_grad_iteration(x, data['loss_vars'], title='Effective loss (variance)')
    plot_y_per_grad_iteration(x, data['mean_fhts'], title='MFHT')

    # get backup policies
    iterations = np.arange(0, args.n_grad_iterations + args.backup_freq, args.backup_freq)[::2]

    env = env.unwrapped
    if env.d <= 2:
        means, stds = agent.get_means_and_stds(data, iterations)

    # plot policy
    if env.d == 1:
        colors, labels = get_colors_and_labels(iterations, iter_str='Grad. iter.')
        plot_det_policies_1d(env, means, sol_hjb.u_opt, colors=colors, labels=labels, loc='upper left')

if __name__ == '__main__':
    main()
