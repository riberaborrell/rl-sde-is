import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments_2d import DoubleWellStoppingTime2D
from rl_sde_is.plots import *
from rl_sde_is.reinforce_deterministic_vectorized import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    return parser

def load_backup_model(data, it=0):
    try:
        load_model(data['model'], data['rel_dir_path'], file_name='model_n-it{}'.format(it))
    except FileNotFoundError as e:
        print('there is no backup for iteration {:d}'.format(it))

def get_policy(env, data, it=None):
    model = data['model']
    if it is not None:
        load_backup_model(data, it)

    states = torch.FloatTensor(env.state_space_h)
    return compute_det_policy_actions(env, model, states)

def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime2D(alpha=args.alpha, beta=args.beta)

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # set action space bounds
    env.action_space_low = 0
    env.action_space_high = 5

    # discretized state space (for plot purposes only)
    env.discretize_state_space(h_state=0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

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
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )

    # do plots
    if not args.plot:
        return

    # plot policy
    policy = get_policy(env, data, it=args.plot_iteration)
    plot_det_policy_2d(env, policy, policy_hjb)

    # plot expected values for each epoch
    plot_expected_returns_with_error_epochs(data['exp_returns'], data['var_returns'])
    plot_time_steps_epochs(data['exp_time_steps'])

    # plot policy l2 error
    plot_det_policy_l2_error_epochs(data['policy_l2_errors'])

    # plot loss function
    plot_loss_epochs(data['losses'])


if __name__ == "__main__":
    main()
