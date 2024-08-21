import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
from gym_sde_is.wrappers.save_episode_trajectory import SaveEpisodeTrajectoryVect

from rl_sde_is.dpg.dpg_utils import DeterministicPolicy, ValueFunction
from rl_sde_is.utils.approximate_methods import evaluate_det_policy_model, \
                                                evaluate_value_function_model, \
                                                train_deterministic_policy_from_hjb
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.path import get_reinforce_det_dir_path, load_data, save_data, \
                                 save_model, load_model
from rl_sde_is.utils.plots import *

def sample_loss(env, model, optimizer, batch_size):

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
    eff_loss = torch.mean(env.girs_det_int - env.returns.detach() * env.girs_stoch_int)
    with torch.no_grad():
        eff_loss_var = torch.var(env.girs_det_int - env.returns * env.girs_stoch_int)

    # reset gradients, compute gradients and update parameters
    optimizer.zero_grad()
    eff_loss.backward()
    optimizer.step()

    return eff_loss, eff_loss_var

def sample_value_loss(env, value, optimizer):

    # compute target value
    with torch.no_grad():

        # value function next
        next_states = np.vstack(env.trajs_states)[1:]
        next_states = np.vstack((next_states, np.zeros((1, env.d))))
        next_states = torch.FloatTensor(next_states)
        v_next = value.forward(next_states)

        # compute target (using target networks)
        done = np.hstack(env.trajs_dones)
        done = torch.tensor(done)
        d = torch.where(done, 1., 0.)
        rewards = np.hstack(env.trajs_rewards)
        rewards = torch.FloatTensor(rewards)
        v_target = rewards + (1. - d) * v_next

    # compute current q-value
    states = np.vstack(env.trajs_states)
    states = torch.FloatTensor(states)
    v_current = value.forward(states)

    # compute loss
    loss = (v_current - v_target).pow(2).mean()

    # reset gradients and update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def reinforce_deterministic(env, gamma, n_layers, d_hidden_layer, batch_size, lr, n_grad_iterations,
                            seed, learn_value, lr_value=None, backup_freq=None, live_plot_freq=None,
                            log_freq=100, policy_opt=None, value_function_opt=None, load=False):

    agent = 'reinforce-deterministic' if not learn_value else 'reinforce-deterministic-value'

    # get dir path
    dir_path = get_reinforce_det_dir_path(
        env,
        agent=agent,
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
    env = RecordEpisodeStatisticsVect(env, batch_size)
    if learn_value:
        env = SaveEpisodeTrajectoryVect(env, batch_size, track_actions=False,
                                        track_rewards=True, track_dones=True)

    # get dimensions of each layer
    d_hidden_layers = [d_hidden_layer for i in range(n_layers-1)]

    # initialize policy model 
    model = DeterministicPolicy(state_dim=env.d, action_dim=env.d,
                                hidden_sizes=d_hidden_layers, activation=nn.Tanh())

    # initialize value function model
    value = ValueFunction(state_dim=env.d, hidden_sizes=d_hidden_layers, activation=nn.Tanh()) \
            if learn_value else None

    # define optimizer/s
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if learn_value:
        value_optimizer = optim.Adam(value.parameters(), lr=lr_value)

    # save algorithm parameters
    data = {
        'gamma': gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'batch_size': batch_size,
        'lr': lr,
        'lr_value': lr_value,
        'n_grad_iterations': n_grad_iterations,
        'seed': seed,
        'learn_value': learn_value,
        'backup_freq': backup_freq,
        'model': model,
        'value': value,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # save model initial parameters
    save_model(model, dir_path, 'model_n-it{}'.format(0))
    if learn_value:
        save_model(value, dir_path, 'value_n-it{}'.format(0))

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(
        eval_freq=1,
        eval_batch_size=batch_size,
        policy_type='det',
        n_grad_iterations=n_grad_iterations,
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
        figs_placeholder = initialize_figures(env, model, value, policy_opt, value_function_opt)

    for i in np.arange(n_grad_iterations+1):

        # start timer
        ct_initial = time.time()

        # compute model based policy effective loss
        eff_loss, eff_loss_var = sample_loss(env, model, optimizer, batch_size)

        if learn_value:
            #value_loss = sample_actor_critic_loss(env, model, optimizer, value, value_optimizer, batch_size)
            value_loss = sample_value_loss(env, value, value_optimizer)


        # end timer
        ct_final = time.time()

        # save and log epoch 
        env.statistics_to_numpy()
        is_stats.save_epoch(i, env, loss=eff_loss.detach().numpy(),
                            loss_var=eff_loss_var.numpy(), ct=ct_final - ct_initial)
        is_stats.log_epoch(i) if i % log_freq == 0 else None

        # backup models and results
        if backup_freq is not None and (i + 1) % backup_freq == 0:
            save_model(model, dir_path, 'model_n-it{}'.format(i + 1))
            if learn_value:
                save_model(value, dir_path, 'value_n-it{}'.format(i + 1))
            stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
            save_data(data | stats_dict, dir_path)

        # update figure
        if live_plot_freq and i % live_plot_freq == 0:
            update_figures(env, model, value, figs_placeholder)

    # add learning results
    stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
    data = data | stats_dict
    save_data(data, dir_path)
    return data

def initialize_figures(env, model, value, policy_opt, value_function_opt):
    # evaluate policy and value function
    policy = evaluate_det_policy_model(env, model).reshape(env.state_space_h.shape)
    if value is not None:
        values = evaluate_value_function_model(env, value).reshape(env.state_space_h.shape[:-1])

    if env.d == 1:
        policy_line = initialize_det_policy_1d_figure(env, policy, policy_opt=policy_opt)
        value_line = initialize_value_function_1d_figure(env, values, value_function_opt) \
                     if value is not None else None
        return policy_line, value_line
    elif env.d == 2:
        policy_quiver = initialize_det_policy_2d_figure(env, policy, policy_opt)
        value_im = initialize_value_function_2d_figure(env, values) \
                   if value is not None else None
        return policy_quiver, value_im


def update_figures(env, model, value, figs_placeholder):

    # evaluate policy and value function
    policy = evaluate_det_policy_model(env, model).reshape(env.state_space_h.shape)
    if value is not None:
        values = evaluate_value_function_model(env, value).reshape(env.state_space_h.shape[:-1])

    if env.d == 1:
        policy_line, value_line = figs_placeholder
        update_det_policy_1d_figure(env, policy, policy_line)
        if value is not None:
            update_value_function_1d_figure(env, values, value_line)

    elif env.d == 2:
        policy_quiver, value_im = figs_placeholder
        update_det_policy_2d_figure(env, policy, policy_quiver)
        if value is not None:
            update_value_function_2d_figure(env, values, value_im)

def load_backup_model(data, i=0):
    try:
        load_model(data['model'], data['dir_path'], file_name='model_n-it{}'.format(i))
        if data['learn_value']:
            load_model(data['value'], data['dir_path'], file_name='value_n-it{}'.format(i))
    except FileNotFoundError as e:
        print('there is no backup for iteration {:d}'.format(i))

def get_policies(env, data, iterations):
    n_iterations = len(iterations)
    policies = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
    for i, it in enumerate(iterations):
        load_backup_model(data, it)
        policies[i] = evaluate_det_policy_model(env, data['model'])
    return policies

def get_value_functions(env, data, iterations):
    n_iterations = len(iterations)
    value_functions = np.empty((n_iterations, env.n_states), dtype=np.float32)
    for i, it in enumerate(iterations):
        load_backup_model(data, it)
        value_functions[i] = evaluate_value_function_model(env, data['value'])
    return value_functions

