from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl_sde_is.discrete.discrete_utils import DQNModel, DuelingCritic
from rl_sde_is.dpg.replay_buffers import ReplayBuffer
from rl_sde_is.utils.tabular_methods import get_epsilons_exp_decay
from rl_sde_is.utils.approximate_methods import *
from rl_sde_is.utils.path import get_dqn_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import *


def select_action(env, model, state, epsilon):

    # sample action randomly
    if np.random.rand() <= epsilon:
        return np.random.choice(np.arange(env.n_actions))

    # choose greedy action
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            return np.argmax(q_values.numpy())

def update_parameters(model, target_model, optimizer, batch, gamma, polyak):

    # unpack tuples in batch
    states = torch.tensor(batch['states'])
    actions = torch.tensor(batch['actions'])
    rewards = torch.tensor(batch['rewards'])
    next_states = torch.tensor(batch['next_states'])
    done = torch.tensor(batch['done'])

    # get batch size
    batch_size = states.shape[0]

    # q value for the given states
    q_vals = model.forward(states)

    with torch.no_grad():

        # max q values of the target modle
        q_vals_next = target_model.forward(next_states)
        max_q_vals_next = torch.max(q_vals_next, axis=1)[0]

        # compute target (using target networks)
        d = torch.where(done, 1., 0.)
        target = rewards + gamma * (1. - d) * max_q_vals_next

    # loss
    loss = (q_vals - target.unsqueeze(dim=1)).pow(2).mean()

    # update critic network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update target network "softly”
    with torch.no_grad():
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1. - polyak))

    return loss.detach().item()


def dqn(env, gamma=1., n_layers=2, d_hidden_layer=32, n_episodes=100, n_steps_lim=1000,
        learning_starts=1000, replay_size=50000, batch_size=512, lr=1e-4, seed=None,
        update_freq=100, polyak=0.995, eps_init=1.0, eps_end=0.01,
        eps_decay=0.995, backup_freq=None, live_plot_freq=None, run_window=10,
        value_function_opt=None, policy_opt=None, load=False):

    # get dir path
    dir_path = get_dqn_dir_path(
        env,
        agent='dqn-episodic',
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
        polyak=polyak,
        batch_size=batch_size,
        lr=lr,
        n_episodes=n_episodes,
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

    # initialize q-value function representation
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    model = DQNModel(state_dim=env.d, n_actions=env.n_actions,
    #model = DuelingCritic(state_dim=env.d, n_actions=env.n_actions,
                     hidden_sizes=hidden_sizes, activation=nn.Tanh())
    target_model = deepcopy(model)

    # set optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=env.d, action_dim=env.n_actions,
                                 size=replay_size, is_action_continuous=False)

    # decaying array of epsilons for the exploration
    epsilons = get_epsilons_exp_decay(n_episodes, eps_init, eps_decay)

    # save algorithm parameters
    data = {
        'gamma' : gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'n_episodes': n_episodes,
        'n_steps_lim': n_steps_lim,
        'replay_size': replay_size,
        'batch_size' : batch_size,
        'lr' : lr,
        'seed': seed,
        'learning_starts': learning_starts,
        'update_freq': update_freq,
        'model': model,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # save models initial parameters
    save_model(model, dir_path, 'model_n-epi{}'.format(0))

    # define list to store results
    returns = np.full(n_episodes, np.nan, dtype=np.float32)
    time_steps = np.full(n_episodes, np.nan, dtype=np.int32)
    cts = np.full(n_episodes, np.nan, dtype=np.float32)

    losses = []

    # total number of time steps
    k_total = 0

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

            # sample action

            # sample action randomly
            if k_total < learning_starts:
                action_idx = np.random.choice(np.arange(env.n_actions))

            # get action following the actor
            else:
                action_idx = select_action(env, model, state, 0.5)

            # env step
            action = env.action_space_h[action_idx]
            next_state, r, done, _, info = env.step(action)

            # store tuple
            replay_buffer.store(state, action, r, next_state, done)

            # time to update
            if k_total >= learning_starts and (k_total + 1) % update_freq == 0:
                for _ in range(update_freq):

                    # sample minibatch of transition uniformlly from the replay buffer
                    batch = replay_buffer.sample_batch(batch_size)

                    # update model parameters
                    loss = update_parameters(model, target_model, optimizer, batch, gamma, polyak)
                    losses.append(loss)

            # save action and reward
            ep_return += (gamma**k) * r

            # update state
            state = next_state

            # update total steps counter
            k_total += 1

        # save episode
        returns[ep] = ep_return
        time_steps[ep] = k
        cts[ep] = time.time() - ct_initial

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
            save_model(model, dir_path, 'model_n-epi{}'.format(ep + 1))

            # save test results
            data['returns'] = returns
            data['time_steps'] = time_steps
            data['cts'] = cts
            data['losses'] = losses
            save_data(data, dir_path)

    # add learning results
    data['returns'] = returns
    data['time_steps'] = time_steps
    data['cts'] = cts
    data['losses'] = np.stack(losses)
    data['replay_states'] = replay_buffer.states[:replay_buffer.size]
    data['replay_actions'] = replay_buffer.actions[:replay_buffer.size]
    save_data(data, dir_path)
    return data

def load_backup_models(data, ep=0):
    model = data['model']
    dir_path = data['dir_path']
    try:
        load_model(model, dir_path, file_name='model_n-epi{}'.format(ep))
    except FileNotFoundError as e:
        print('The episode {:d} has no backup '.format(ep))

