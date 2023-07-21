from copy import deepcopy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import mlp
from rl_sde_is.plots import *
from rl_sde_is.replay_buffers import ReplayBuffer, ReplayMemory
from rl_sde_is.utils_path import *

class DeterministicPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.policy = mlp(self.sizes, activation)
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-3, 5e-3)
                nn.init.uniform_(module.bias, -5e-3, 5e-3)

    def forward(self, state):
        return self.policy.forward(state)

class QValueFunction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim + action_dim] + list(hidden_sizes) + [1]
        self.q = mlp(self.sizes, activation)

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return torch.squeeze(q, axis=-1)

def update_parameters(actor, actor_optimizer, critic, batch, fht=1.):

    # unpack tuples in batch
    states = torch.tensor(batch['states'])

    # get batch size
    batch_size = states.shape[0]

    # freeze q-networks to save computational effort 
    for param in critic.parameters():
        param.requires_grad = False

    # actor loss
    actor_loss = - fht * critic.forward(states, actor.forward(states)).mean()

    # reset actor gradients
    actor_optimizer.zero_grad()

    #update actor network
    actor_loss.backward()
    actor_optimizer.step()

    # unfreeze q-network to save computational effort 
    for param in critic.parameters():
        param.requires_grad = True


def get_action(env, actor, state, noise_scale=0):

    # forward pass
    action = actor.forward(torch.FloatTensor(state)).detach().numpy()

    # add noise
    action += noise_scale * np.random.randn(env.action_space_dim)

    # clipp such that it lies in the valid action range
    action = np.clip(action, env.action_space_low, env.action_space_high)
    return action


def dpg(env, gamma=1.00, d_hidden_layer=32, n_layers=3, n_steps_episode_lim=int(1e5),
        batch_size=int(1e3), lr_type='adaptive', lr_actor=1e-4, seed=None, n_iterations=100,
        test_freq_iterations=100, test_batch_size=1000, backup_freq_iterations=None,
        value_function_opt=None, policy_opt=None, load=False, l2_error=True, live_plot=False):

    action_limit = 5

    # get dir path
    rel_dir_path = get_dpg_dir_path(
        env,
        agent='dpg',
        gamma=gamma,
        d_hidden_layer=d_hidden_layer,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_type=lr_type,
        n_iterations=n_iterations,
        seed=seed,
    )

    # load results
    if load:
        data = load_data(rel_dir_path)
        return data

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # dimensions of the state and action space
    d_state_space = env.state_space_dim
    d_action_space = env.action_space_dim

    # initialize replay buffer
    replay_buffer = ReplayBuffer(
        size=int(1e7), state_dim=d_state_space, action_dim=d_action_space,
    )

    # initialize actor representations
    actor_hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    actor = DeterministicPolicy(
        state_dim=d_state_space,
        action_dim=d_action_space,
        hidden_sizes=actor_hidden_sizes,
        activation=nn.Tanh(),
    )
    actor_target = deepcopy(actor)

    # initialize critic representations
    critic_hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    critic = QValueFunction(
        state_dim=d_state_space,
        action_dim=d_action_space,
        hidden_sizes=critic_hidden_sizes,
        activation=nn.Tanh()
    )
    critic = train_critic_from_dp(env, critic, value_function_opt, policy_opt, load=True)

    # set optimizers
    #actor_optimizer = optim.SGD(actor.parameters(), lr=lr_actor)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)

    # save algorithm parameters
    data = {
        'gamma' : gamma,
        'd_hidden_layer': d_hidden_layer,
        'n_layers': n_layers,
        'action_limit': action_limit,
        'n_steps_episode_lim': n_steps_episode_lim,
        'batch_size' : batch_size,
        'lr_actor' : lr_actor,
        'lr_type' : lr_type,
        'n_iterations': n_iterations,
        'seed': seed,
        'test_freq_iterations': test_freq_iterations,
        'test_batch_size': test_batch_size,
        'actor': actor,
        'critic': critic,
        'rel_dir_path': rel_dir_path,
    }
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-it{}'.format(0))

    # ct per iterations
    cts = np.empty(n_iterations, dtype=np.float32)
    cts.fill(np.nan)

    # test mean, variance and mean length of the returns and l2 error after each epoch
    test_mean_returns = np.empty((0), dtype=np.float32)
    test_var_returns = np.empty((0), dtype=np.float32)
    test_mean_lengths = np.empty((0), dtype=np.float32)
    test_policy_l2_errors = np.empty((0), dtype=np.float32)

    # test initial actor model
    test_mean_ret, test_var_ret, test_mean_len, test_policy_l2_error \
            = test_policy_vectorized(env, actor, batch_size=test_batch_size,
                                     policy_opt=policy_opt)
    test_mean_returns = np.append(test_mean_returns, test_mean_ret)
    test_var_returns = np.append(test_var_returns, test_var_ret)
    test_mean_lengths = np.append(test_mean_lengths, test_mean_len)
    test_policy_l2_errors = np.append(test_policy_l2_errors, test_policy_l2_error)

    msg = 'it: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
          'test mean time steps: {:2.2f}, test policy l2 error: {:.2e}'.format(
              0,
              test_mean_ret,
              test_var_ret,
              test_mean_len,
              test_policy_l2_error,
          )
    print(msg)

    # get initial state
    state_init = env.state_init.copy()

    # total number of time steps
    k_total = 0

    # initialize figures if plot:
    if live_plot and env.d == 1:
        policy = compute_table_det_policy_1d(env, actor)
        line_actor = initialize_det_policy_1d_figure(env, policy, policy_opt)
    elif live_plot and env.d == 2:
        pass

    for it in range(n_iterations):

        # start timer
        ct_initial = time.time()

        # sample trajectories
        sample_trajectories_buffer_vectorized(env, actor, replay_buffer,
                                              batch_size, n_steps_episode_lim)

        # sample minibatch of transition uniformlly from the replay buffer
        batch = replay_buffer.sample_batch(int(1e3))

        # constant learning type
        if lr_type == 'constant':

            # update actor parameters
            update_parameters(actor, actor_optimizer, critic, batch)

        elif lr_type == 'adaptive':

            # estimate fht
            fht = replay_buffer.estimate_episode_length()

            # update actor parameters
            update_parameters(actor, actor_optimizer, critic, batch, fht)

        # reset replay buffer
        replay_buffer.reset()

        # end timer
        ct_final = time.time()

        # save episode
        cts[it] = ct_final - ct_initial

        #print('it: {:3d}, fht: {:2.2f}'.format(it+1, fht))

        # test actor model
        if (it + 1) % test_freq_iterations == 0:

            test_mean_ret, test_var_ret, test_mean_len, test_policy_l2_error \
                    = test_policy_vectorized(env, actor, batch_size=test_batch_size,
                                             policy_opt=policy_opt)
            test_mean_returns = np.append(test_mean_returns, test_mean_ret)
            test_var_returns = np.append(test_var_returns, test_var_ret)
            test_mean_lengths = np.append(test_mean_lengths, test_mean_len)
            test_policy_l2_errors = np.append(test_policy_l2_errors, test_policy_l2_error)

            msg = 'it: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
                  'test mean time steps: {:2.2f}, test policy l2 error: {:.2e}, ct: {:2.3f}'.format(
                it + 1,
                test_mean_ret,
                test_var_ret,
                test_mean_len,
                test_policy_l2_error,
                cts[it],
            )
            print(msg)

        # backup models and results
        if backup_freq_iterations is not None and (it + 1) % backup_freq_iterations == 0:

            # save actor and critic models
            save_model(actor, rel_dir_path, 'actor_n-epi{}'.format(it + 1))

            # save test results
            data['returns'] = returns
            data['time_steps'] = time_steps
            data['cts'] = cts
            data['test_mean_returns'] = test_mean_returns
            data['test_var_returns'] = test_var_returns
            data['test_mean_lengths'] = test_mean_lengths
            data['test_policy_l2_errors'] = test_policy_l2_errors

            save_data(data, rel_dir_path)

        # update plots
        if live_plot and (it + 1) % 1 == 0:
            if env.d == 1:
                policy = compute_table_det_policy_1d(env, actor)
                update_det_policy_1d_figure(env, policy, line_actor)

            elif env.d == 2:
                pass

    data['cts'] = cts
    data['test_mean_returns'] = test_mean_returns
    data['test_var_returns'] = test_var_returns
    data['test_mean_lengths'] = test_mean_lengths
    data['test_policy_l2_errors'] = test_policy_l2_errors
    save_data(data, rel_dir_path)
    return data


def load_backup_models(data, it=0):
    actor = data['actor']
    critic = data['critic']
    rel_dir_path = data['rel_dir_path']
    try:
        load_model(actor, rel_dir_path, file_name='actor_n-it{}'.format(ep))
    except FileNotFoundError as e:
        print('there is no backup after episode {:d}'.format(ep))

