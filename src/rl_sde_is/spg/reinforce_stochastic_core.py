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
from rl_sde_is.utils.path import get_reinforce_stoch_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import initialize_gaussian_policy_1d_figure, update_gaussian_policy_1d_figure

def sample_loss_random_time(env, policy, optimizer, batch_size, return_type):
    ''' Sample and compute loss function corresponding to the policy gradient with
        random time expectation. Also update the policy parameters.
    '''

    # initialization
    state, _ = env.reset(batch_size=batch_size)

    # terminal state flag
    done = np.full((batch_size,), False)
    while not done.all():

        # sample action
        state_torch = torch.FloatTensor(state)
        with torch.no_grad():
            action, _ = policy.sample_action(state_torch)

        # env step
        state, _, _, truncated, _ = env.step_vect(action)
        done = np.logical_or(env.been_terminated, truncated)

    # compute log probs
    states = torch.FloatTensor(np.vstack(env.trajs_states))
    actions = torch.FloatTensor(np.vstack(env.trajs_actions))
    _, log_probs = policy.forward(states, actions)

    # compute returns
    returns = []
    for i in range(batch_size):

        # compute initial returns
        if return_type == 'initial-return':
            returns.append(np.full(env.lengths[i], env.returns[i]))

        # compute n-step returns
        else: # return_type == 'n-return'
            returns.append(cumsum(env.trajs_rewards[i]))

    returns = torch.FloatTensor(np.hstack(returns))

    # calculate loss
    phi = - log_probs * returns

    # loss and loss variance
    loss = phi.sum() / batch_size
    with torch.no_grad():
        loss_var = phi.var().numpy()

    # reset gradients, compute gradients and update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, loss_var


def sample_loss_on_policy(env, policy, optimizer, batch_size, return_type, mini_batch_size,
                          memory_size, estimate_mfht):

    # initialize memory
    memory = ReplayMemoryReturn(size=memory_size, state_dim=env.d, action_dim=env.d,
                                return_type=return_type)

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

    # compute returns
    returns = []
    for i in range(batch_size):

        # compute initial returns
        if return_type == 'initial-return':
            returns.append(np.full(env.lengths[i], env.returns[i]))

        # compute n-step returns
        else: # return_type == 'n-return'
            returns.append(cumsum(env.trajs_rewards[i]))

    returns = torch.FloatTensor(np.hstack(returns))

    # store experiences in memory
    states = np.vstack(env.trajs_states)
    actions = np.vstack(env.trajs_actions)
    if return_type == 'initial-return':
        memory.store_vectorized(states, actions, initial_returns=returns)
    else:
        memory.store_vectorized(states, actions, n_returns=returns)

    # sample batch of experiences from memory
    batch = memory.sample_batch(mini_batch_size)
    _, log_probs = policy.forward(batch['states'], batch['actions'])
    mfht = env.lengths.mean() if estimate_mfht else 1

    # calculate loss
    returns = batch['n-returns'] if return_type == 'n-return' else batch['initial-returns']
    phi = - (log_probs * returns)
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

    # reset memory
    memory.reset()

    return loss, loss_var

def reinforce_stochastic(env, expectation_type, return_type, gamma, policy_type,
                         n_layers, d_hidden_layer, theta_init, policy_noise, batch_size, lr,
                         n_grad_iterations, learn_value, estimate_mfht=None, mini_batch_size=None,
                         memory_size=int(1e6), seed=None,
                         backup_freq=None, live_plot_freq=None, log_freq=100,
                         policy_opt=None, value_function_opt=None, load=False):

    # get dir path
    dir_path = get_reinforce_stoch_dir_path(
        env,
        agent='reinforce-stoch-{}'.format(expectation_type),
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
        theta_init=theta_init,
        policy_type=policy_type,
        policy_noise=policy_noise,
        return_type=return_type,
        estimate_mfht=estimate_mfht,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        lr=lr,
        n_grad_iterations=n_grad_iterations,
        learn_value=learn_value,
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

    # save states, action and rewards
    if return_type == 'n-return':
        env = SaveEpisodeTrajectoryVect(env, batch_size, track_rewards=True)

    # save states and actions 
    else: #return_type == 'initial-return':
        env = SaveEpisodeTrajectoryVect(env, batch_size)

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
        if expectation_type == 'random-time':
            loss, loss_var = sample_loss_random_time(env, policy, optimizer, batch_size, return_type)
        else: #expectation_type == 'on-policy':
            loss, loss_var = sample_loss_on_policy(env, policy, optimizer, batch_size, return_type,
                                         mini_batch_size, memory_size, estimate_mfht)

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

