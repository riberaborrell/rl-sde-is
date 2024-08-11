from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl_sde_is.dpg.replay_buffers import ReplayBuffer
from rl_sde_is.utils.models import mlp
from rl_sde_is.utils.approximate_methods import *
from rl_sde_is.utils.path import get_naf_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import *

class NAFModel(nn.Module):
    ''' The normalized advantage function (NAF) model is a neural network that represents
        the value function, the policy and a state-dependent positive definite matrix P.
    '''
    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super(NAFModel, self).__init__()

        self.action_dim = action_dim
        sizes = [state_dim] + list(hidden_sizes)
        self.net = mlp(sizes, activation, activation)
        self.value_layer = nn.Linear(hidden_sizes[-1], 1)
        self.policy_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.L_layer = nn.Linear(hidden_sizes[-1], action_dim * action_dim)

    def forward(self, state):
        y = self.net.forward(state)
        value = self.value_layer(y)
        policy = self.policy_layer(y)
        L = self.L_layer(y)
        L = L.view(-1, self.action_dim, self.action_dim)
        P = torch.bmm(L, L.transpose(1, 2))
        return value.squeeze(), policy, P


def compute_q_value(value_function, policy, P, action):
    ''' Computes the q-value function in terms of the value function and the advantage function
    '''
    action_diff = (action - policy).unsqueeze(-1)
    advantage_function = - 0.5 * torch.bmm(
        action_diff.transpose(1, 2),
        torch.bmm(P, action_diff)
    ).squeeze()
    return value_function + advantage_function


def update_parameters(model, target_model, optimizer, batch, gamma, polyak):

    # unpack tuples in batch
    states = torch.tensor(batch['states'])
    actions = torch.tensor(batch['actions'])
    rewards = torch.tensor(batch['rewards'])
    next_states = torch.tensor(batch['next_states'])
    done = torch.tensor(batch['done'])

    # get batch size
    batch_size = states.shape[0]

    # compute target q-value
    with torch.no_grad():

        # value function next
        value_next, _, _ = target_model.forward(next_states)

        # compute target (using target networks)
        d = torch.where(done, 1., 0.)
        q_target = rewards + gamma * (1. - d) * value_next.squeeze()

    # compute current q-value
    value, mu, P = model.forward(states)
    q_current = compute_q_value(value, mu, P, actions)

    # compute loss
    loss = (q_current - q_target).pow(2).mean()

    # update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update target model "softly”
    with torch.no_grad():
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1. - polyak))

def get_action(env, model, state, noise_scale, action_limit):

    # forward pass
    with torch.no_grad():
        _, action, _ = model.forward(torch.FloatTensor(state))

    # add noise
    action = action.numpy() + noise_scale * np.random.randn(env.action_space.shape[0])

    # clipp such that it lies in the valid action range
    action = np.clip(action, -action_limit, action_limit)
    return action


def naf(env, gamma=1., n_layers=3, d_hidden_layer=32,
        n_episodes=100, n_steps_lim=1000, learning_starts=1000,
        expl_noise_init=1.0, expl_noise_decay=1., replay_size=50000,
        batch_size=1000, lr=1e-4, seed=None,
        polyak=0.95, action_limit=None,
        backup_freq=None, live_plot_freq=None, run_window=10,
        value_function_opt=None, policy_opt=None, load=False):

    # get dir path
    dir_path = get_naf_dir_path(
        env,
        agent='naf',
        gamma=gamma,
        d_hidden_layer=d_hidden_layer,
        expl_noise_init=expl_noise_init,
        action_limit=action_limit,
        polyak=polyak,
        batch_size=batch_size,
        lr=lr,
        n_episodes=n_episodes,
        n_steps_lim=n_steps_lim,
        seed=seed,
    )

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


    # initialize naf model representation
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    model = NAFModel(state_dim=env.d, action_dim=env.d,
                     hidden_sizes=hidden_sizes, activation=nn.Tanh())
    target_model = deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=env.d, action_dim=env.d, size=replay_size)

    # save algorithm parameters
    data = {
        'gamma' : gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'n_episodes': n_episodes,
        'n_steps_lim': n_steps_lim,
        'expl_noise_init': expl_noise_init,
        'expl_noise_decay': expl_noise_decay,
        'replay_size': replay_size,
        'replay_states': replay_buffer.states[:replay_buffer.size],
        'replay_actions': replay_buffer.actions[:replay_buffer.size],
        'batch_size' : batch_size,
        'lr' : lr,
        'seed': seed,
        'learning_starts': learning_starts,
        'action_limit': action_limit,
        'polyak': polyak,
        'model': model,
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

    # set noise scale parameters
    expl_noises = np.array([
        expl_noise_init * (expl_noise_decay ** ep)
        for ep in np.arange(n_episodes)
    ])

    # initialize figures if plot
    if live_plot_freq:
        if env.d == 1:
            tuple_fig_replay = initialize_replay_buffer_1d_figure(env, replay_buffer)
            lines_returns = initialize_return_and_time_steps_figures(env, n_episodes)
        elif env.d == 2:
            lines_returns = initialize_return_and_time_steps_figures(env, n_episodes)

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
                action = env.action_space.sample()

            # get action following the actor
            else:
                action = get_action(env, model, state, expl_noises[ep], action_limit)

            # env step
            next_state, r, done, _, info = env.step(action)

            # store tuple
            replay_buffer.store(state, action, r, next_state, done)

            # time to update
            if k_total >= learning_starts:

                # sample minibatch of transition uniformlly from the replay buffer
                batch = replay_buffer.sample_batch(batch_size)

                # update model
                update_parameters(model, target_model, optimizer, batch, gamma, polyak)

            # save action and reward
            ep_return += (gamma**k) * r

            # update state
            state = next_state

            # update total steps counter
            k_total += 1

        # end timer
        ct_final = time.time()

        # save statistics
        returns[ep] = ep_return
        time_steps[ep] = k
        cts[ep] = ct_final - ct_initial

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
            save_data(data, dir_path)

        # update plots
        if live_plot_freq and (ep + 1) % live_plot_freq == 0:
            if env.d == 1:
                #update_1d_figures(env, actor, critic1, lines_actor_critic)
                update_replay_buffer_1d_figure(env, replay_buffer, tuple_fig_replay)
                update_return_and_time_steps_figures(env, returns[:ep], time_steps[:ep], lines_returns)

            elif env.d == 2:
                #update_2d_figures(env, actor, Q_policy)
                update_return_and_time_steps_figures(env, returns[:ep], time_steps[:ep], lines_returns)

    # add learning results
    data['returns'] = returns
    data['time_steps'] = time_steps
    data['cts'] = cts
    data['replay_states'] = replay_buffer.states[:replay_buffer.size]
    data['replay_actions'] = replay_buffer.actions[:replay_buffer.size]

    save_data(data, dir_path)
    return data

def load_backup_models(data, ep=0):
    model = data['model']
    dir_path = data['dir_path']
    try:
        load_model(model, dir_path, file_name='model_n-ep{}'.format(ep))
    except FileNotFoundError as e:
        print('The episode {:d} has no backup '.format(ep))

def eval_model_state_space(env, data):
    model = data['model']
    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    with torch.no_grad():
        value_function, policy, P = model.forward(state_space_h)
    return value_function.numpy().squeeze(), policy.numpy()

def get_value_functions_and_policies(env, data, episodes):
    n_episodes = len(episodes)
    value_functions = np.empty((n_episodes, env.n_states), dtype=np.float32)
    policies = np.empty((n_episodes, env.n_states, env.d), dtype=np.float32)
    for i, ep in enumerate(episodes):
        load_backup_models(data, ep)
        value_functions[i], policy = eval_model_state_space(env, data)
        policies[i] = policy[0]
    return value_functions, policies
