import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
from gym_sde_is.wrappers.save_episode_trajectory import SaveEpisodeTrajectoryVect

from rl_sde_is.spg.spg_utils import GaussianPolicyConstantCov, GaussianPolicyLearntCov
from rl_sde_is.spg.replay_memories import ReplayMemoryReturn
from rl_sde_is.utils.approximate_methods import evaluate_stoch_policy_model, \
                                                train_stochastic_policy_from_hjb
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.numeric import cumsum_numpy as cumsum
from rl_sde_is.utils.path import get_reinforce_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import initialize_gaussian_policy_1d_figure, update_gaussian_policy_1d_figure

def sample_loss_random_time_initial_return(env, policy, optimizer, batch_size):
    ''' Sample and compute loss function corresponding to the policy gradient with
        random time expectation and initial return. Also update the policy parameters.
    '''

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
        idx = ~env.been_terminated | env.new_terminated
        log_probs[idx] = log_probs[idx] + log_probs_n[idx]

    # calculate loss
    returns_torch = torch.FloatTensor(env.returns)
    eff_loss = - torch.mean(log_probs * returns_torch)
    eff_loss_var = - np.var(log_probs.detach().numpy() * env.returns)

    # reset gradients, compute gradients and update parameters
    optimizer.zero_grad()
    eff_loss.backward()
    optimizer.step()

    return eff_loss, eff_loss_var

def sample_loss_random_time_n_step_return(env, policy, optimizer, batch_size):
    ''' Sample and compute loss function corresponding to the policy gradient with
        random time expectation and n-step return. Also update the policy parameters.
    '''

    # initialization
    state, _ = env.reset(batch_size=batch_size)

    # terminal state flag
    done = np.full((batch_size,), False)
    while not done.all():

        # sample action
        state_torch = torch.FloatTensor(state)
        action, _ = policy.sample_action(state_torch)

        # env step
        state, _, _, truncated, _ = env.step_vect(action)
        done = np.logical_or(env.been_terminated, truncated)

    # initialize tensor
    phi = torch.empty(batch_size)

    for i in range(batch_size):

        # calculate after n step returns
        n_returns = torch.FloatTensor(cumsum(env.trajs_rewards[i]).copy())

        # calculate log probs
        states = torch.FloatTensor(env.trajs_states[i])
        actions = torch.FloatTensor(env.trajs_actions[i])
        _, log_probs = policy.forward(states, actions)

        phi[i] = - torch.dot(log_probs, n_returns)

    # calculate loss
    eff_loss = phi.mean()
    with torch.no_grad():
        eff_loss_var = phi.var().numpy()

    # reset gradients, compute gradients and update parameters
    optimizer.zero_grad()
    eff_loss.backward()
    optimizer.step()

    return eff_loss, eff_loss_var

def sample_loss_on_policy_n_step_return(env, policy, optimizer, batch_size, mini_batch_size, estimate_mfht):

    # initialize memory
    memory = ReplayMemoryReturn(size=int(1e6), state_dim=env.d, action_dim=env.d)

    # initialization
    state, _ = env.reset(batch_size=batch_size)

    # terminal state flag
    done = np.full((batch_size,), False)
    while not done.all():

        # sample action
        state_torch = torch.FloatTensor(state)
        action, _ = policy.sample_action(state_torch)

        # env step
        state, _, _, truncated, _ = env.step_vect(action)
        done = np.logical_or(env.been_terminated, truncated)

    for i in range(batch_size):

        # calculate after n step returns
        n_returns = cumsum(env.trajs_rewards[i]).copy()

        # calculate log probs
        states = env.trajs_states[i]
        actions = env.trajs_actions[i]

        # store experiences in memory
        memory.store_vectorized(states, actions, n_returns)

    # sample batch of experiences from memory
    batch = memory.sample_batch(mini_batch_size)
    states = torch.FloatTensor(batch['states'])
    actions = torch.FloatTensor(batch['actions'])
    n_returns = torch.FloatTensor(batch['n_returns'])
    _, log_probs = policy.forward(states, actions)
    mfht = env.lengths.mean() if estimate_mfht else 1

    # calculate loss
    phi = - (log_probs * n_returns)
    loss = phi.mean()
    with torch.no_grad():
        loss_var = phi.var().numpy()

    # reset and compute actor gradients
    optimizer.zero_grad()
    loss.backward()

    # scale learning rate
    optimizer.param_groups[0]['lr'] *= mfht

    #update parameters
    optimizer.step()

    # re-scale learning rate back
    optimizer.param_groups[0]['lr'] /= mfht

    # reset replay buffer
    memory.reset()

    return loss, loss_var

def reinforce_stochastic(env, algorithm_type, expectation_type, gamma, policy_type,
                         n_layers, d_hidden_layer, theta_init, policy_noise, estimate_mfht,
                         batch_size, mini_batch_size, lr, n_grad_iterations, seed=None,
                         backup_freq=None, policy_opt=None, load=False, live_plot_freq=None):

    # get dir path
    dir_path = get_reinforce_dir_path(
        env,
        agent='reinforce-{}-{}'.format(expectation_type, algorithm_type),
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
        theta_init=theta_init,
        policy_type=policy_type,
        policy_noise=policy_noise,
        estimate_mfht=estimate_mfht,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
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

    # record states and action from the trajectories wrapper
    if algorithm_type == 'n-return':
        env = SaveEpisodeTrajectoryVect(env, batch_size, track_rewards=True)

    # initialize model and optimizer
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]

    if policy_type == 'const-cov':
        policy = GaussianPolicyConstantCov(state_dim=env.d, action_dim=env.d,
                                           hidden_sizes=hidden_sizes, activation=nn.Tanh(),
                                           std=policy_noise)
    else:
        policy = GaussianPolicyLearntCov(
            state_dim=env.d, action_dim=env.d, hidden_sizes=hidden_sizes,
            activation=nn.Tanh(), std_init=policy_noise,
        )
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # train params to fit hjb solution
    if theta_init == 'hjb':
        train_stochastic_policy_from_hjb(env, policy, policy_opt, load=True)

    # save algorithm parameters
    data = {
        'gamma': gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'policy_type': policy_type,
        'policy_noise': policy_noise,
        'batch_size' : batch_size,
        'mini_batch_size' : mini_batch_size,
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
        policy_type='stoch',
        n_grad_iterations=n_grad_iterations,
        track_loss=True,
        track_ct=True,
    )
    keys_chosen = [
        'max_lengths', 'mean_fhts', 'var_fhts',
        'mean_returns', 'var_returns',
        'losses', 'loss_vars',
        'cts',
    ]

    if live_plot_freq and env.d == 1:
        mean, sigma = evaluate_stoch_policy_model(env, policy)
        lines = initialize_gaussian_policy_1d_figure(env, mean, sigma, policy_opt=policy_opt)

    for i in np.arange(n_grad_iterations+1):

        # start timer
        ct_initial = time.time()

        # sample loss function
        if algorithm_type == 'initial-return' and expectation_type == 'random-time':
            loss, loss_var = sample_loss_random_time_initial_return(env, policy, optimizer, batch_size)
        elif algorithm_type == 'n-return' and expectation_type == 'random-time':
            loss, loss_var = sample_loss_random_time_n_step_return(env, policy, optimizer, batch_size)
        elif algorithm_type == 'initial-return' and expectation_type == 'on-policy':
            raise NotImplementedError('On-policy initial return not implemented')
        elif algorithm_type == 'n-return' and expectation_type == 'on-policy':
            loss, loss_var = sample_loss_on_policy_n_step_return(env, policy, optimizer, batch_size,
                                                                 mini_batch_size, estimate_mfht)

        # end timer
        ct_final = time.time()

        # save and log epoch 
        env.statistics_to_numpy()
        is_stats.save_epoch(i, env, loss=loss.detach().numpy(),
                            loss_var=loss_var, ct=ct_final - ct_initial)
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

def get_means_and_stds(env, data, iterations):

    n_iterations = len(iterations)
    means = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
    stds = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
    for i, it in enumerate(iterations):
        load_backup_model(data, it)
        mean, std = evaluate_stoch_policy_model(env, data['policy'])
        means[i] = mean.reshape(env.n_states, env.d)
        stds[i] = std#.reshape(env.n_states, env.d)
    return means, stds

