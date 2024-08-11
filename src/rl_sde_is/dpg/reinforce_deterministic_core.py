import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
from gym_sde_is.utils.sde import compute_is_functional

from rl_sde_is.dpg.dpg_utils import DeterministicPolicy
from rl_sde_is.utils.approximate_methods import compute_det_policy_actions, evaluate_det_policy_model
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.path import get_reinforce_det_dir_path, load_data, save_data, \
                                 save_model, load_model
from rl_sde_is.utils.numeric import logistic_torch
from rl_sde_is.utils.plots import *

def sample_loss(env, model, batch_size):

    # initialization
    state, _ = env.reset(batch_size=batch_size, is_torch=True)

    # terminal state flag
    done = np.full((batch_size,), False)
    while not done.all():

        # sample action
        action = model.forward(state)

        # env step
        state, _, _, truncated, _ = env.step_vect_torch(action)
        done = np.logical_or(env.been_terminated, truncated)

    # calculate loss
    eff_loss = torch.mean(-env.returns - env.returns.detach() * env.girs_stoch_int)
    with torch.no_grad():
        eff_loss_var = torch.var(-env.returns - env.returns * env.girs_stoch_int)

    return eff_loss, eff_loss_var


def reinforce_deterministic(env, gamma=1., n_layers=2, d_hidden_layer=32, batch_size=1000,
                            lr=1e-3, n_grad_iterations=100, seed=None, backup_freq=None,
                            live_plot_freq=None, policy_opt=None, track_l2_error=False, load=False):

    # get dir path
    dir_path = get_reinforce_det_dir_path(
        env,
        agent='reinforce-deterministic',
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
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
        np.random.seed(seed)
        torch.manual_seed(seed)

    # vectorized environment
    env = RecordEpisodeStatisticsVect(env, batch_size, track_l2_error)

    # get dimensions of each layer
    d_hidden_layers = [d_hidden_layer for i in range(n_layers-1)]

    # initialize nn model 
    model = DeterministicPolicy(state_dim=env.d, action_dim=env.d,
                                hidden_sizes=d_hidden_layers, activation=nn.Tanh())

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # save algorithm parameters
    data = {
        'gamma': gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'batch_size': batch_size,
        'lr': lr,
        'n_grad_iterations': n_grad_iterations,
        'seed': seed,
        'backup_freq': backup_freq,
        'model': model,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # save model initial parameters
    save_model(model, dir_path, 'model_n-it{}'.format(0))

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(
        eval_freq=1,
        eval_batch_size=batch_size,
        n_grad_iterations=n_grad_iterations,
        track_l2_error=track_l2_error,
        track_loss=True,
        track_ct=True,
    )
    keys_chosen = [
        'max_lengths', 'mean_fhts', 'var_fhts',
        'mean_returns', 'var_returns',
        'mean_I_us', 'var_I_us', 're_I_us',
        'losses', 'loss_vars',
        'cts',
    ]

    # initialize live figures
    if live_plot_freq:
        if env.d == 1:
            policy_line = initialize_1d_figures(env, model, policy_opt)
        elif env.d == 2:
            Q_policy = initialize_2d_figures(env, model, policy_opt)

    for i in np.arange(n_grad_iterations+1):

        # start timer
        ct_initial = time.time()

        # compute effective loss
        eff_loss, eff_loss_var = sample_loss(env, model, batch_size)

        # reset gradients and update parameters
        optimizer.zero_grad()
        eff_loss.backward()
        optimizer.step()

        # end timer
        ct_final = time.time()

        # save and log epoch 
        env.statistics_to_numpy()
        l2_errors = env.l2_errors if track_l2_error else None
        is_functional = compute_is_functional(
            env.girs_stoch_int,
            env.running_rewards,
            env.terminal_rewards,
        )
        is_stats.save_epoch(
            i, env.lengths, env.lengths*env.dt, env.returns,
            is_functional=is_functional, l2_errors=l2_errors,
            loss=eff_loss.detach().numpy(), loss_var=eff_loss_var.numpy(),
            ct=ct_final - ct_initial,
        )
        is_stats.log_epoch(i)

        # backup models and results
        if backup_freq is not None and (i + 1) % backup_freq == 0:
            save_model(model, dir_path, 'model_n-it{}'.format(i + 1))
            stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
            save_data(data | stats_dict, dir_path)

        # update figure
        if live_plot_freq and i % live_plot_freq == 0:
            if env.d == 1:
                update_1d_figures(env, model, policy_line)
            elif env.d == 2:
                update_2d_figures(env, model, Q_policy)

    # add learning results
    stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
    data = data | stats_dict
    save_data(data, dir_path)
    return data

def initialize_1d_figures(env, model, policy_opt):

    # hjb control
    if policy_opt is None:
        policy_opt_plot = np.empty_like(env.state_space_h)
        policy_opt_plot.fill(np.nan)
    else:
        policy_opt_plot = policy_opt

    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    initial_policy = compute_det_policy_actions(env, model, state_space_h).squeeze()
    policy_line = initialize_det_policy_1d_figure(env, initial_policy, policy_opt=policy_opt_plot)

    return policy_line

def update_1d_figures(env, model, policy_line):
    states = torch.FloatTensor(env.state_space_h)
    policy = compute_det_policy_actions(env, model, states)
    update_det_policy_1d_figure(env, policy, policy_line)

def initialize_2d_figures(env, model, policy_hjb):
    states = torch.FloatTensor(env.state_space_h)
    initial_policy = compute_det_policy_actions(env, model, states)
    Q_policy = initialize_det_policy_2d_figure(env, initial_policy, policy_hjb)
    return Q_policy

def update_2d_figures(env, model, Q_policy):
    states = torch.FloatTensor(env.state_space_h)
    policy = compute_det_policy_actions(env, model, states)
    update_det_policy_2d_figure(env, policy, Q_policy)

def load_backup_model(data, i=0):
    try:
        load_model(data['model'], data['dir_path'], file_name='model_n-it{}'.format(i))
    except FileNotFoundError as e:
        print('there is no backup for iteration {:d}'.format(i))

def get_policies(env, data, iterations):

    n_iterations = len(iterations)
    policies = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
    for i, it in enumerate(iterations):
        load_backup_model(data, it)
        policies[i] = evaluate_det_policy_model(env, data['model']).reshape(env.n_states, env.d)
    return policies

