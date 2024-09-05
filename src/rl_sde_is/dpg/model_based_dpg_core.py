import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl_sde_is.dpg.dpg_utils import DeterministicPolicy, ValueFunction
from rl_sde_is.dpg.replay_memories import ReplayMemoryModelBasedDPG as ReplayMemory
from rl_sde_is.utils.approximate_methods import *
from rl_sde_is.utils.numeric import dot_vect
from rl_sde_is.utils.path import get_model_based_dpg_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import *


def update_parameters(env, model, optimizer, batch, gamma, z_estimate):

    # unpack tuples in batch
    states = batch['states']
    dbts = batch['dbts']
    returns = batch['returns']
    dones = batch['dones']
    d = torch.where(dones, 1., 0.)

    # get batch size
    batch_size = states.shape[0]

    # sample action
    actions = model.forward(states)

    # compute loss
    loss = torch.mean(
        gamma * (1 - d) * (
            0.5 * torch.linalg.norm(actions, axis=1).pow(2) * env.dt
          - returns * dot_vect(dbts, actions)
        )
    )

    # reset and compute gradients
    optimizer.zero_grad()
    loss.backward()

    # scale learning rate
    optimizer.param_groups[0]['lr'] *= z_estimate

    #update parameters
    optimizer.step()

    # re-scale learning rate back
    optimizer.param_groups[0]['lr'] /= z_estimate

    return loss.detach().item()



def model_based_dpg(env, return_type, gamma, n_layers, d_hidden_layer, theta_init, batch_size, lr, n_episodes, n_steps_lim,
                    seed, learning_starts, replay_size, estimate_z, learn_value, lr_value=None,
                    backup_freq=None, live_plot_freq=None, log_freq=100, run_window=10,
                    value_function_opt=None, policy_opt=None, load=False):

    # get dir path
    dir_path = get_model_based_dpg_dir_path(
        env,
        agent='model-based-dpg',
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
        theta_init=theta_init,
        return_type=return_type,
        estimate_z=estimate_z,
        batch_size=batch_size,
        lr=lr,
        n_episodes=n_episodes,
        n_steps_lim=n_steps_lim,
        replay_size=replay_size,
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

    # record statistics wrapper
    env = RecordEpisodeStatistics(env, n_episodes)

    # initialize actor representations
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    model = DeterministicPolicy(state_dim=env.d, action_dim=env.d,
                                hidden_sizes=hidden_sizes, activation=nn.Tanh())

    # initialize value function model
    value = ValueFunction(state_dim=env.d, hidden_sizes=d_hidden_layers, activation=nn.Tanh()) \
            if learn_value else None

    # set optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if learn_value:
        value_optimizer = optim.Adam(value.parameters(), lr=lr_value)

    # initialize replay memory
    replay_memory = ReplayMemory(state_dim=env.d, size=replay_size)

    # save algorithm parameters
    data = {
        'gamma' : gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'batch_size' : batch_size,
        'lr' : lr,
        'n_episodes': n_episodes,
        'n_steps_lim': n_steps_lim,
        'seed': seed,
        'learn_value': learn_value,
        'replay_size': replay_size,
        'learning_starts': learning_starts,
        'model': model,
        'value': value,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # save models initial parameters
    save_model(model, dir_path, 'model_n-ep{}'.format(0))

    # preallocate arrays
    returns = np.full(n_episodes, np.nan, dtype=np.float32)
    time_steps = np.full(n_episodes, np.nan, dtype=np.int32)
    cts = np.full(n_episodes, np.nan, dtype=np.float32)

    # total number of time steps
    k_total = 0

    # initialize figures if plot
    if live_plot_freq:
        figs_placeholder = initialize_figures(env, n_episodes, model,
                                              replay_memory, policy_opt)

    # sample trajectories
    for ep in range(n_episodes):

        # start timer
        ct_initial = time.time()

        # initialization
        state, _ = env.reset()

        # reset trajectory return
        ep_return = 0

        # terminal state flag
        done = False

        # sample trajectory
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # select action
            with torch.no_grad():
                action = model.forward(torch.FloatTensor(state)).numpy()

            # env step
            next_state, r, done, _, info = env.step(action)

            # store tuple
            replay_memory.store(state=state, dbt=info['dbt'], reward=r, done=done)

            # save action and reward
            ep_return += (gamma**k) * r

            # update state
            state = next_state

            # update total steps counter
            k_total += 1

        replay_memory.compute_returns()

        # save statistics
        returns[ep] = ep_return
        time_steps[ep] = k
        cts[ep] = time.time() - ct_initial

        # time to update
        if k_total >= learning_starts:

            for _ in range(k):

                # sample minibatch of experiences uniformlly from the replay memory
                batch = replay_memory.sample_batch(batch_size)

                # estimate the mean length
                z_estimate = replay_memory.estimate_mean_episode_length() if estimate_z else 1

                # update model parameters
                loss = update_parameters(env, model, optimizer, batch, gamma, z_estimate)


        if ep % log_freq == 0:
            msg = 'ep.: {:2d}, return: {:.3e} (avg. {:.2e}, max. {:.2e}), time steps: {:.3e}, ct: {:.3f}'.format(
                ep,
                returns[ep],
                np.mean(returns[:ep+1][-run_window:]),
                np.max(returns[:ep+1][-run_window:]),
                time_steps[ep],
                cts[ep],
            )
            print(msg)

        # backup models and results
        if backup_freq is not None and (ep + 1) % backup_freq == 0:

            # save actor and critic models
            save_model(model, dir_path, 'model_n-ep{}'.format(ep + 1))

            # save test results
            data['returns'] = returns
            data['time_steps'] = time_steps
            data['cts'] = cts
            data['replay_states'] = replay_memory.states[:replay_memory.size]

            save_data(data, dir_path)

        # update plots
        if live_plot_freq and (ep + 1) % live_plot_freq == 0:
            update_figures(env, model, replay_memory, returns, time_steps, figs_placeholder)

    # add learning results
    data['returns'] = returns
    data['time_steps'] = time_steps
    data['cts'] = cts
    data['replay_states'] = replay_memory.states[:replay_memory.size]

    save_data(data, dir_path)
    return data

def initialize_figures(env, n_episodes, model, replay_memory, policy_opt):

    # return and time steps
    return_lines = initialize_return_and_time_steps_figures(env, n_episodes)
    policy = evaluate_det_policy_model(env, model).reshape(env.state_space_h.shape)
    if env.d == 1:
        policy_line = initialize_det_policy_1d_figure(env, policy, policy_opt=policy_opt)
        return return_lines, policy_line

    elif env.d == 2:
        policy_quiver = initialize_det_policy_2d_figure(env, policy, policy_opt)
        return return_lines, policy_quiver

    return return_lines

def update_figures(env, model, replay_memory, returns, time_steps, figs_placeholder):
    policy = evaluate_det_policy_model(env, model).reshape(env.state_space_h.shape)
    if env.d == 1:
        return_lines, policy_line = figs_placeholder
        update_det_policy_1d_figure(env, policy, policy_line)

    elif env.d == 2:
        return_lines, policy_quiver = figs_placeholder
        update_det_policy_2d_figure(env, policy, policy_quiver)
    else:
        return_lines = figs_placeholder

    # return and time steps
    update_return_and_time_steps_figures(env, returns, time_steps, return_lines)

def load_backup_models(data, ep=0):
    try:
        load_model(data['model'], data['dir_path'], file_name='model_n-ep{}'.format(ep))
        if data['learn_value']:
            load_model(data['value'], data['dir_path'], file_name='value_n-ep{}'.format(ep))
        return True
    except FileNotFoundError as e:
        print('There is no backup for episode {:d}'.format(ep))
        return False

def get_policies(env, data, episodes):
    n_episodes = len(episodes)
    policies = np.empty((n_episodes, env.n_states, env.d), dtype=np.float32)
    for i, ep in enumerate(episodes):
        load_backup_models(data, ep)
        policies[i] = evaluate_det_policy_model(env, data['model'])
    return policies

def get_value_functions(env, data, episodes):
    n_episodes = len(episodes)
    value_functions = np.empty((n_episodes, env.n_states), dtype=np.float32)
    for i, ep in enumerate(episodes):
        load_backup_models(data, ep)
        value_functions[i] = evaluate_value_function_model(env, data['value'])
    return value_functions

