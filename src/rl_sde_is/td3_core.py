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
from rl_sde_is.replay_buffers import ContinuousReplayBuffer as ReplayBuffer
from rl_sde_is.utils_path import *

class DeterministicPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_limit):
        super().__init__()
        self.sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.policy = mlp(self.sizes, activation)
        self.action_limit = action_limit
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-3, 5e-3)
                nn.init.uniform_(module.bias, -5e-3, 5e-3)

    def forward(self, state):
        return self.policy.forward(state)
        #action = self.policy.forward(state)
        #return self.action_limit * torch.tanh(action)

class QValueFunction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim + action_dim] + list(hidden_sizes) + [1]
        self.q = mlp(self.sizes, activation)
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-4, 5e-4)
                nn.init.uniform_(module.bias, -5e-4, 5e-4)

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return torch.squeeze(q, axis=-1)

def update_parameters(actor, actor_target, actor_optimizer,
                      critic1, critic_target1, critic2, critic_target2, critic_optimizer,
                      batch, gamma, policy_delay, timer,
                      noise_clip=0., action_limit=5, target_noise=0., polyak=0.995):

    # unpack tuples in batch
    states = torch.tensor(batch['state'])
    next_states = torch.tensor(batch['next_state'])
    actions = torch.tensor(batch['act'])
    rewards = torch.tensor(batch['rew'])
    done = torch.tensor(batch['done'])

    # get batch size
    batch_size = states.shape[0]

    # 1) run 1 gradient descent step for Q1 and Q2 (critics)

    # reset critic gradients
    critic_optimizer.zero_grad()

    # q value for the given pairs of states and actions (forward pass of the critic network)
    q_vals1 = critic1.forward(states, actions)
    q_vals2 = critic2.forward(states, actions)

    with torch.no_grad():

        # next actions following the target actor
        next_actions = actor_target.forward(next_states).detach()

        # target policy smoothing
        epsilon = torch.randn_like(next_actions) * target_noise
        #epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
        next_actions_smoothed = next_actions + epsilon
        next_actions_smoothed = torch.clamp(next_actions_smoothed, -action_limit, action_limit)

        # q value for the corresponding next pair of states and actions (using target networks)
        q_vals_next1 = critic_target1.forward(next_states, next_actions_smoothed)
        q_vals_next2 = critic_target2.forward(next_states, next_actions_smoothed)

        # compute target (using target networks)
        d = torch.where(done, 1., 0.)
        q_vals_next = torch.min(q_vals_next1, q_vals_next2)
        target = rewards + gamma * (1. - d) * q_vals_next

    # critic loss
    critic_loss1 = ((q_vals1 - target)**2).mean()
    critic_loss2 = ((q_vals2 - target)**2).mean()
    critic_loss = critic_loss1 + critic_loss2

    # update critic network
    critic_loss.backward()
    critic_optimizer.step()

    # 2) run 1 gradient descent step for mu (actor)

     # Possibly update pi and target networks
    if timer % policy_delay == 0:

        # freeze q-networks to save computational effort 
        for param in critic1.parameters():
                param.requires_grad = False
        for param in critic2.parameters():
                param.requires_grad = False

        # reset actor gradients
        actor_optimizer.zero_grad()

        # actor loss
        actor_loss = - critic1.forward(states, actor.forward(states)).mean()

        # update actor network
        actor_loss.backward()
        actor_optimizer.step()

        # unfreeze q-network to save computational effort 
        for param in critic1.parameters():
                param.requires_grad = True
        for param in critic2.parameters():
                param.requires_grad = True

        # update actor and critic target networks "softly”
        with torch.no_grad():
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(target_param.data * polyak + param.data * (1. - polyak))

            for param, target_param in zip(critic1.parameters(), critic_target1.parameters()):
                target_param.data.copy_(target_param.data * polyak + param.data * (1. - polyak))

            for param, target_param in zip(critic2.parameters(), critic_target2.parameters()):
                target_param.data.copy_(target_param.data * polyak + param.data * (1. - polyak))

        #return actor_loss.detach().item(), critic_loss.detach().item()

def get_action(env, actor, state, noise_scale=0):

    # forward pass
    action = actor.forward(torch.FloatTensor(state)).detach().numpy()

    # add noise
    action += noise_scale * np.random.randn(env.action_space_dim)

    # clipp such that it lies in the valid action range
    action = np.clip(action, env.action_space_low, env.action_space_high)
    return action


def td3_episodic(env, gamma=0.99, d_hidden_layer=32, n_layers=3, action_limit=5,
                 n_episodes=100, n_steps_episode_lim=1000,
                 start_steps=0, expl_noise_init=0.1, expl_noise_decay=1., replay_size=50000,
                 batch_size=1000, lr_actor=1e-4, lr_critic=1e-4, seed=None,
                 update_after=5000, update_every=100, policy_delay=50, target_noise=0.2, polyak=0.95,
                 test_freq_episodes=100, test_batch_size=1000, backup_freq_episodes=None,
                 value_function_opt=None, policy_opt=None, load=False, live_plot=False):

    # get dir path
    rel_dir_path = get_td3_dir_path(
        env,
        agent='td3-episodic',
        gamma=gamma,
        d_hidden_layer=d_hidden_layer,
        expl_noise_init=expl_noise_init,
        target_noise=target_noise,
        policy_delay=policy_delay,
        polyak=polyak,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        n_episodes=n_episodes,
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

    # initialize actor representations
    actor_hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    actor = DeterministicPolicy(
        state_dim=d_state_space,
        action_dim=d_action_space,
        hidden_sizes=actor_hidden_sizes,
        activation=nn.Tanh(),
        action_limit=action_limit,
    )
    actor_target = deepcopy(actor)

    # initialize critic representations
    critic_hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    critic1 = QValueFunction(
        state_dim=d_state_space,
        action_dim=d_action_space,
        hidden_sizes=critic_hidden_sizes,
        activation=nn.Tanh()
    )
    critic_target1 = deepcopy(critic1)
    critic2 = QValueFunction(
        state_dim=d_state_space,
        action_dim=d_action_space,
        hidden_sizes=critic_hidden_sizes,
        activation=nn.Tanh()
    )
    critic_target2 = deepcopy(critic2)

    # set optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_params = list(critic1.parameters()) + list(critic2.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
    #critic_optimizer = optim.Adam(critic_params, lr=lr_critic, weight_decay=1e-2)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=d_state_space, action_dim=d_action_space,
                                 size=replay_size)

    # save algorithm parameters
    data = {
        'gamma' : gamma,
        'd_hidden_layer': d_hidden_layer,
        'n_layers': n_layers,
        'action_limit': action_limit,
        'n_episodes': n_episodes,
        'n_steps_episode_lim': n_steps_episode_lim,
        'start_steps': start_steps,
        'expl_noise_init': expl_noise_init,
        'expl_noise_decay': expl_noise_decay,
        'replay_size': replay_size,
        'replay_states': replay_buffer.state_buf[:replay_buffer.size],
        'replay_actions': replay_buffer.act_buf[:replay_buffer.size],
        'batch_size' : batch_size,
        'lr_actor' : lr_actor,
        'lr_critic' : lr_critic,
        'seed': seed,
        'update_after': update_after,
        'update_every': update_every,
        'policy_delay': policy_delay,
        'target_noise': target_noise,
        'polyak': polyak,
        'test_freq_episodes': test_freq_episodes,
        'test_batch_size': test_batch_size,
        'actor': actor,
        'critic1': critic1,
        'critic2': critic2,
        'rel_dir_path': rel_dir_path,
    }
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-epi{}'.format(0))
    save_model(critic1, rel_dir_path, 'critic1_n-epi{}'.format(0))
    save_model(critic2, rel_dir_path, 'critic2_n-epi{}'.format(0))

    # preallocate arrays

    # returns, time steps and ct per episode
    returns = np.empty(n_episodes, dtype=np.float32)
    returns.fill(np.nan)
    time_steps = np.zeros(n_episodes, dtype=np.int32)
    cts = np.empty(n_episodes, dtype=np.float32)
    cts.fill(np.nan)

    # test mean, variance and mean length of the returns and l2 error after each epoch
    test_mean_returns = np.empty((0), dtype=np.float32)
    test_var_returns = np.empty((0), dtype=np.float32)
    test_mean_lengths = np.empty((0), dtype=np.float32)
    test_policy_l2_errors = np.empty((0), dtype=np.float32)

    # test initial actor model
    test_mean_ret, test_var_ret, test_mean_len, test_policy_l2_error \
            = test_policy_vectorized(env, actor, batch_size=test_batch_size,
                                     control_hjb=policy_opt)
    test_mean_returns = np.append(test_mean_returns, test_mean_ret)
    test_var_returns = np.append(test_var_returns, test_var_ret)
    test_mean_lengths = np.append(test_mean_lengths, test_mean_len)
    test_policy_l2_errors = np.append(test_policy_l2_errors, test_policy_l2_error)

    msg = 'ep: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
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

    # set noise scale parameters
    expl_noises = np.array([
        expl_noise_init * (expl_noise_decay ** ep)
        for ep in np.arange(n_episodes)
    ])

    # initialize figures if plot:
    if live_plot and env.d == 1:
        lines_actor_critic = initialize_1d_figures(env, actor, critic1, value_function_opt, policy_opt)
        tuple_fig_replay = initialize_replay_buffer_1d_figure(env, replay_buffer)
        lines_returns = initialize_return_and_time_steps_figures(env, n_episodes)
    elif live_plot and env.d == 2:
        Q_policy = initialize_2d_figures(env, actor, policy_opt)
        lines_returns = initialize_return_and_time_steps_figures(env, n_episodes)


    # sample trajectories
    for ep in range(n_episodes):
        print(ep)

        # start timer
        ct_initial = time.time()

        # initialization
        state = env.reset()

        # reset trajectory return
        ep_return = 0

        # terminal state flag
        done = False

        # sample trajectory
        for k in np.arange(n_steps_episode_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # sample action

            # sample action randomly
            if k_total < start_steps:
                action = env.sample_action(batch_size=1)

            # get action following the actor
            else:
                action = get_action(env, actor, state, expl_noises[ep])

            # env step
            next_state, r, done, _ = env.step(state, action)
            #print(k, state, action, r, next_state)

            # store tuple
            replay_buffer.store(state, action, r, next_state, done)

            # time to update
            if k_total >= update_after and (k_total + 1) % update_every == 0:

                for l in range(update_every):

                    # sample minibatch of transition uniformlly from the replay buffer
                    batch = replay_buffer.sample_batch(batch_size)

                    # update actor and critic parameters
                    update_parameters(
                        actor, actor_target, actor_optimizer,
                        critic1, critic_target1, critic2, critic_target2, critic_optimizer,
                        batch, gamma, policy_delay, l, target_noise=target_noise, polyak=polyak,
                    )

            # save action and reward
            ep_return += (gamma**k) * r

            # update state
            state = next_state

            # update total steps counter
            k_total += 1

        # end timer
        ct_final = time.time()

        # save episode
        returns[ep] = ep_return
        time_steps[ep] = k
        cts[ep] = ct_final - ct_initial

        # test actor model
        if (ep + 1) % test_freq_episodes == 0:

            test_mean_ret, test_var_ret, test_mean_len, test_policy_l2_error \
                    = test_policy_vectorized(env, actor, batch_size=test_batch_size,
                                             control_hjb=policy_opt)
            test_mean_returns = np.append(test_mean_returns, test_mean_ret)
            test_var_returns = np.append(test_var_returns, test_var_ret)
            test_mean_lengths = np.append(test_mean_lengths, test_mean_len)
            test_policy_l2_errors = np.append(test_policy_l2_errors, test_policy_l2_error)

            msg = 'ep: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
                  'test mean time steps: {:2.2f}, test policy l2 error: {:.2e}'.format(
                ep + 1,
                test_mean_ret,
                test_var_ret,
                test_mean_len,
                test_policy_l2_error,
            )
            print(msg)

        # backup models and results
        if backup_freq_episodes is not None and (ep + 1) % backup_freq_episodes == 0:

            # save actor and critic models
            save_model(actor, rel_dir_path, 'actor_n-epi{}'.format(ep + 1))
            save_model(critic1, rel_dir_path, 'critic1_n-epi{}'.format(ep + 1))
            save_model(critic2, rel_dir_path, 'critic2_n-epi{}'.format(ep + 1))

            # save test results
            data['returns'] = returns
            data['time_steps'] = time_steps
            data['cts'] = cts
            data['test_mean_returns'] = test_mean_returns
            data['test_var_returns'] = test_var_returns
            data['test_mean_lengths'] = test_mean_lengths
            data['test_policy_l2_errors'] = test_policy_l2_errors
            data['replay_states'] = replay_buffer.state_buf[:replay_buffer.size]
            data['replay_actions'] = replay_buffer.act_buf[:replay_buffer.size]

            save_data(data, rel_dir_path)

        # update plots
        if live_plot and (ep + 1) % 1 == 0:
            if env.d == 1:
                update_1d_figures(env, actor, critic1, lines_actor_critic)
                update_replay_buffer_1d_figure(env, replay_buffer, tuple_fig_replay)
                update_return_and_time_steps_figures(env, returns[:ep], time_steps[:ep], lines_returns)

            elif env.d == 2:
                update_2d_figures(env, actor, Q_policy)
                update_return_and_time_steps_figures(env, returns[:ep], time_steps[:ep], lines_returns)

    data['returns'] = returns
    data['time_steps'] = time_steps
    data['cts'] = cts
    data['test_mean_returns'] = test_mean_returns
    data['test_var_returns'] = test_var_returns
    data['test_mean_lengths'] = test_mean_lengths
    data['test_policy_l2_errors'] = test_policy_l2_errors
    data['replay_states'] = replay_buffer.state_buf[:replay_buffer.size]
    data['replay_actions'] = replay_buffer.act_buf[:replay_buffer.size]
    save_data(data, rel_dir_path)
    return data

def initialize_1d_figures(env, actor, critic1, value_function_opt, policy_opt):
    q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic1)
    v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic1)
    lines = initialize_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                            a_table, policy_actor, policy_critic,
                                            value_function_opt, policy_opt)
    return lines

def update_1d_figures(env, actor, critic1, lines):
    q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic1)
    v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic1)
    update_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                a_table, policy_actor, policy_critic, lines)

def initialize_2d_figures(env, actor, policy_hjb):
    states = torch.FloatTensor(env.state_space_h)
    initial_policy = compute_det_policy_actions(env, actor, states)
    Q_policy = initialize_det_policy_2d_figure(env, initial_policy, policy_hjb)
    return Q_policy

def update_2d_figures(env, actor, Q_policy):
    states = torch.FloatTensor(env.state_space_h)
    policy = compute_det_policy_actions(env, actor, states)
    update_det_policy_2d_figure(env, policy, Q_policy)

def load_backup_models(data, ep=0):
    actor = data['actor']
    critic1 = data['critic1']
    critic2 = data['critic2']
    rel_dir_path = data['rel_dir_path']
    try:
        load_model(actor, rel_dir_path, file_name='actor_n-epi{}'.format(ep))
        load_model(critic1, rel_dir_path, file_name='critic1_n-epi{}'.format(ep))
        load_model(critic2, rel_dir_path, file_name='critic2_n-epi{}'.format(ep))
    except FileNotFoundError as e:
        print('there is no backup after episode {:d}'.format(ep))

