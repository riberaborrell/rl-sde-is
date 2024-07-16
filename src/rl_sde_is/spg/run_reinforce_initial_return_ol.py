import gymnasium as gym
import gym_sde_is
from gym_sde_is.utils.sde import compute_is_functional
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.spg.spg_utils import *
from rl_sde_is.approximate_methods import compute_table_stoch_policy_1d
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.path import get_reinforce_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import *


def reinforce(env, gamma=1., n_layers=3, d_hidden_layer=32, policy_type='const-cov', policy_noise=1.,
              lr=1e-4, batch_size=1, n_iterations=100, seed=None,
              backup_freq=None, policy_opt=None, load=False, live_plot_freq=None):

    # get dir path
    dir_path = get_reinforce_dir_path(
        env,
        agent='reinforce-initial-return',
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
        policy_type=policy_type,
        policy_noise=policy_noise,
        batch_size=batch_size,
        lr=lr,
        n_iterations=n_iterations,
        seed=seed,
    )

    # load results
    if load:
        return load_data(dir_path)

    # save algorithm parameters
    data = {
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'policy_type': policy_type,
        'batch_size' : batch_size,
        'n_iterations': n_iterations,
        'lr' : lr,
        'seed': seed,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initialize model and optimizer
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]

    if policy_type == 'const-cov':
        policy = GaussianPolicyConstantCov(state_dim=env.d, action_dim=env.d,
                                           hidden_sizes=hidden_sizes, activation=nn.Tanh(),
                                           std=policy_noise)
    else:
        policy = GaussianPolicyLearntCov(
            state_dim=env.d, action_dim=env.d, hidden_sizes=hidden_sizes, activation=nn.Tanh(),
        )
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    save_model(policy, dir_path, 'policy_it{}'.format(0))

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(eval_freq=1, eval_batch_size=batch_size, n_iterations=n_iterations,
                            track_loss=True, track_l2_error=env.track_l2_error)

    if live_plot_freq and env.d == 1:
        mean, sigma = compute_table_stoch_policy_1d(env, policy)
        lines = initialize_gaussian_policy_1d_figure(env, mean, sigma, policy_opt=policy_opt)

    for i in np.arange(n_iterations):

        # reset gradients
        optimizer.zero_grad()

        # initialization
        state, _ = env.reset(batch_size=batch_size)

        # preallocate log probs
        log_probs = torch.zeros(batch_size)

        # terminal state flag
        done = np.full((batch_size,), False)
        while not done.all():

            # sample action
            state_torch = torch.FloatTensor(state)
            action, _ = policy.sample_action(state_torch)
            action_torch = torch.FloatTensor(action)
            _, log_probs_n = policy.forward(state_torch, action_torch)

            # env step
            state, _, _, truncated, _ = env.step_vect(action)
            done = np.logical_or(env.been_terminated, truncated)

            # save log probs 
            idx = ~env.been_terminated
            log_probs[idx] = log_probs[idx] + log_probs_n[idx]

        # calculate loss
        returns_torch = torch.FloatTensor(env.returns)
        eff_loss = - torch.mean(log_probs * returns_torch)
        eff_loss_var = - np.var(log_probs.detach().numpy() * env.returns)

        # calculate gradients
        eff_loss.backward()

        # update coefficients
        optimizer.step()

        # save is stats
        l2_errors = env.l2_errors if env.track_l2_error else None
        is_functional = compute_is_functional(env.girs_stoch_int,
                                              env.running_rewards, env.terminal_rewards)
        is_stats.save_epoch(i, env.lengths, env.lengths*env.dt, env.returns,
                            is_functional=is_functional, loss=eff_loss.detach().numpy(),
                            l2_errors=l2_errors)
        is_stats.log_epoch(i)

        # backup model
        if backup_freq and (i + 1) % backup_freq== 0:
            save_model(policy, dir_path, 'policy_it{}'.format(i + 1))

        # update plots
        if live_plot_freq and env.d == 1 and (i + 1) % live_plot_freq == 0:
            mean, sigma = compute_table_stoch_policy_1d(env, policy)
            update_gaussian_policy_1d_figure(env, mean, sigma, lines)

    data['policy'] = policy
    save_data(data, dir_path)
    return data

def load_backup_models(data, i=0):
    try:
        load_model(data['policy'], data['dir_path'], file_name='policy_it{}'.format(i))
    except FileNotFoundError as e:
        print('The iteration {:d} has no backup '.format(i))

def get_means(env, data, iterations):

    Nx = env.n_states
    means = np.empty((0, Nx, env.d), dtype=np.float32)

    for i in iterations:
        load_backup_models(data, i)
        mean, _ = compute_table_stoch_policy_1d(env, data['policy'])
        means = np.vstack((means, mean.reshape(1, Nx, env.d)))
    return means

def main():
    parser = get_base_parser()
    parser.description = 'Run reinforce with initial returns for the sde importance sampling environment'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        alpha=np.array(args.alpha),
        beta=args.beta,
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatisticsVect(env, args.batch_size, args.track_l2_error)

    # discretize state and action space (plot purposes only)
    h_coarse = 0.05
    env.discretize_state_space(h_state=h_coarse)
    env.discretize_action_space(h_action=h_coarse)

    # compute corresponding beta
    beta = 2 / (env.dt + env.sigma**2)
    sigma = np.sqrt(2 / beta)

    # get hjb solver
    sol_hjb = env.get_hjb_solver(beta=beta)
    sol_hjb.coarse_solution(h_coarse)
    policy_opt = sol_hjb.u_opt * env.sigma / sigma

    # run reinforce with initial return
    data = reinforce(
        env,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        policy_type=args.policy_type,
        policy_noise=args.policy_noise,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        n_iterations=args.n_iterations,
        backup_freq=args.backup_freq,
        policy_opt=policy_opt,
        load=args.load,
        live_plot_freq=args.live_plot_freq,
    )

    # plot results
    if not args.plot:
        return

    # get backup policies
    iterations = np.arange(0, args.n_iterations + args.backup_freq, args.backup_freq)
    means = get_means(env, data, iterations[::10])

    # plot avg returns and mfht
    x = np.arange(data['n_iterations'])
    plot_y_per_x(x, data['objectives'], title='Objective function', xlabel='Iterations')
    plot_y_per_x(x, data['losses'], title='Effective loss', xlabel='Iterations')
    plot_y_per_x(x, data['loss_vars'], title='Effective loss', xlabel='Iterations')
    plot_y_per_x(x, data['mfhts'], title='MFHT', xlabel='Iterations')


    # plot policy
    if env.d == 1:
        plot_det_policies_1d(env, means, policy_opt)

        #mean, sigma = compute_table_stoch_policy_1d(env, data['policy'])
        #plot_gaussian_stoch_policy_1d(env, mean.squeeze(), sigma.squeeze(), policy_opt)
        #plot_gaussian_stoch_policy_1d(env, mean.squeeze(), sigma.squeeze(), sol_hjb.u_opt.squeeze())

if __name__ == '__main__':
    main()
