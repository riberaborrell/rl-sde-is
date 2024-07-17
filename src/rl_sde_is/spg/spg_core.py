import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import compute_table_stoch_policy_1d
from rl_sde_is.spg.spg_utils import *
from rl_sde_is.utils.path import get_reinforce_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import initialize_gaussian_policy_1d_figure, update_gaussian_policy_1d_figure

def reinforce_initial_return(env, gamma=1., n_layers=3, d_hidden_layer=32, policy_type='const-cov',
                             policy_noise=1., lr=1e-4, batch_size=1, n_iterations=100, seed=None,
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
        'policy_noise': policy_noise,
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

    # preallocate iteration arrays
    objectives = np.empty(n_iterations, dtype=np.float32)
    losses = np.empty(n_iterations, dtype=np.float32)
    loss_vars = np.empty(n_iterations, dtype=np.float32)
    mfhts = np.empty(n_iterations, dtype=np.float32)

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

        # save stats
        objectives[i] = env.returns.mean()
        losses[i] = eff_loss.detach().numpy()
        loss_vars[i] = eff_loss_var
        mfhts[i] = (env.lengths * env.dt).mean()

        # log iteration
        print('it: {}, objective: {:.3f}, eff loss: {:.2e}, eff loss var: {:.2e}, mfht: {:.2e}' \
              ''.format(i, objectives[i], losses[i], loss_vars[i], mfhts[i]))

        # backup model
        if backup_freq and (i + 1) % backup_freq== 0:
            save_model(policy, dir_path, 'policy_it{}'.format(i + 1))

        # update plots
        if live_plot_freq and env.d == 1 and i % live_plot_freq == 0:
            mean, sigma = compute_table_stoch_policy_1d(env, policy)
            update_gaussian_policy_1d_figure(env, mean, sigma, lines)

    data['objectives'] = objectives
    data['losses'] = losses
    data['loss_vars'] = loss_vars
    data['mfhts'] = mfhts
    data['policy'] = policy
    save_data(data, dir_path)
    return data

def load_backup_model(data, i=0):
    try:
        load_model(data['policy'], data['dir_path'], file_name='policy_it{}'.format(i))
    except FileNotFoundError as e:
        print('The iteration {:d} has no backup '.format(i))

def get_means(env, data, iterations):

    Nx = env.n_states
    means = np.empty((0, Nx, env.d), dtype=np.float32)

    for i in iterations:
        load_backup_model(data, i)
        mean, _ = compute_table_stoch_policy_1d(env, data['policy'])
        means = np.vstack((means, mean.reshape(1, Nx, env.d)))
    return means

