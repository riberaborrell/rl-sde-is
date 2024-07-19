import time

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
from gym_sde_is.utils.sde import compute_is_functional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import evaluate_stoch_policy_model
from rl_sde_is.spg.spg_utils import *
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.path import get_reinforce_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import initialize_gaussian_policy_1d_figure, update_gaussian_policy_1d_figure

def reinforce_initial_return(env, gamma=1., n_layers=3, d_hidden_layer=32, policy_type='const-cov',
                             policy_noise=1., lr=1e-4, batch_size=1, n_grad_iterations=100, seed=None,
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
        n_grad_iterations=n_grad_iterations,
        seed=seed,
    )

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # vectorized environment
    env = RecordEpisodeStatisticsVect(env, batch_size)

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

    # save algorithm parameters
    data = {
        'gamma': gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'policy_type': policy_type,
        'policy_noise': policy_noise,
        'batch_size' : batch_size,
        'lr' : lr,
        'n_grad_iterations': n_grad_iterations,
        'seed': seed,
        'backup_freq': backup_freq,
        'policy': policy,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # save model initial parameters
    save_model(policy, dir_path, 'policy_n-it{}'.format(0))

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(
        eval_freq=1,
        eval_batch_size=batch_size,
        n_grad_iterations=n_grad_iterations,
        track_loss=True,
        track_ct=True,
    )
    keys_chosen = [
        'mean_fhts', 'var_fhts',
        'mean_returns', 'var_returns',
        'mean_I_us', 'var_I_us', 're_I_us',
        'losses', 'loss_vars',
        'cts',
    ]

    if live_plot_freq and env.d == 1:
        mean, sigma = evaluate_stoch_policy_model(env, policy)
        lines = initialize_gaussian_policy_1d_figure(env, mean, sigma, policy_opt=policy_opt)

    for i in np.arange(n_grad_iterations):

        # start timer
        ct_initial = time.time()

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

        # end timer
        ct_final = time.time()

        # save and log epoch 
        env.statistics_to_numpy()
        is_functional = compute_is_functional(
            env.girs_stoch_int,
            env.running_rewards,
            env.terminal_rewards,
        )
        is_stats.save_epoch(
            i, env.lengths, env.lengths*env.dt, env.returns,
            is_functional=is_functional,
            loss=eff_loss.detach().numpy(), loss_var=eff_loss_var,
            ct=ct_final - ct_initial,
        )
        is_stats.log_epoch(i)

        # backup models and results
        if backup_freq and (i + 1) % backup_freq== 0:
            save_model(policy, dir_path, 'policy_n-it{}'.format(i + 1))
            stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
            save_data(data | stats_dict, dir_path)

        # update plots
        if live_plot_freq and env.d == 1 and i % live_plot_freq == 0:
            mean, sigma = evaluate_stoch_policy_model(env, policy)
            update_gaussian_policy_1d_figure(env, mean, sigma, lines)

    stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
    data = data | stats_dict
    save_data(data, dir_path)
    return data

def load_backup_model(data, i=0):
    try:
        load_model(data['policy'], data['dir_path'], file_name='policy_n-it{}'.format(i))
    except FileNotFoundError as e:
        print('The iteration {:d} has no backup '.format(i))

def get_means(env, data, iterations):

    n_iterations = len(iterations)
    means = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
    for i, it in enumerate(iterations):
        load_backup_model(data, it)
        mean, _ = evaluate_stoch_policy_model(env, data['policy'])
        means[i] = mean.reshape(env.n_states, env.d)
    return means

