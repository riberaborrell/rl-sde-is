import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.reinforce_deterministic_core import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    return parser

def get_policy(env, data, it=None):
    model = data['model']
    if it is not None:
        load_backup_model(data, it)

    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    with torch.no_grad():
        policy = model.forward(state_space_h).numpy().squeeze()
    return policy

def get_policies(env, data, iterations):

    Nx = env.n_states
    policies = np.empty((0, Nx), dtype=np.float32)

    for it in iterations:
        load_backup_model(data, it)
        policies = np.vstack((policies, get_policy(env, data).reshape(1, Nx)))

    return policies

def get_backup_policies(env, data):
    n_iterations = data['n_iterations']
    backup_freq_iterations = data['backup_freq_iterations']

    Nx = env.n_states
    policies = np.empty((0, Nx), dtype=np.float32)

    for i in range(data['n_iterations']):
        if i == 0:
            load_backup_model(data, 0)
            policies = np.vstack((policies, get_policy(env, data).reshape(1, Nx)))

        #elif (i + 1) % backup_freq_iterations == 0:
        elif (i + 1) % 100 == 0:
            load_backup_model(data, i+1)
            policies = np.vstack((policies, get_policy(env, data).reshape(1, Nx)))

    return policies


def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # set action space bounds
    env.set_action_space_bounds()

    # discretized state space (for plot purposes only)
    env.discretize_state_space(h_state=0.05)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    control_hjb = np.expand_dims(sol_hjb.u_opt, axis=1)

    # run reinforve algorithm with a deterministic policy
    data = reinforce(
        env,
        gamma=args.gamma,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=args.n_iterations,
        backup_freq_iterations=args.backup_freq_iterations,
        seed=args.seed,
        test_freq_iterations=args.test_freq_iterations,
        test_batch_size=args.test_batch_size,
        policy_opt=control_hjb,
        load=args.load,
        test=args.test,
        live_plot=args.live_plot,
    )

    # do plots
    if not args.plot:
        return

    # plot policy
    policy = get_policy(env, data, it=args.plot_iteration)
    plot_det_policy_1d(env, policy, sol_hjb.u_opt)

    policies = get_backup_policies(env, data)
    plot_det_policies_1d(env, policies, sol_hjb.u_opt)
    #plot_det_policies_black_and_white(env, policies, sol_hjb.u_opt)

    # plot expected values for each epoch
    plot_expected_returns_with_error_epochs(data['exp_returns'], data['var_returns'])
    plot_time_steps_epochs(data['exp_time_steps'])

    # plot loss function
    plot_loss_epochs(data['losses'])

    # plot policy l2 error
    if 'test_policy_l2_errors' in data:
        plot_det_policy_l2_error_epochs(data['test_policy_l2_errors'])

if __name__ == "__main__":
    main()
