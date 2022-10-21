from copy import deepcopy

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

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.policy = mlp(sizes, activation)

    def forward(self, state):
        return self.policy.forward(state)

class QValueFunction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp(sizes=[state_dim + action_dim]+list(hidden_sizes)+[1], activation=activation)

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return torch.squeeze(q, axis=-1)

def update_parameters(actor, actor_target, actor_optimizer,
                      critic1, critic_target1, critic2, critic_target2, critic_optimizer,
                      batch, gamma, policy_delay, timer,
                      rho=0.95, noise_clip=0.1, act_limit=5, target_noise=.1):

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
        epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
        next_actions_smoothed = next_actions + epsilon
        next_actions_smoothed = torch.clamp(next_actions_smoothed, -act_limit, act_limit)

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
                target_param.data.copy_(target_param.data * rho + param.data * (1. - rho))

            for param, target_param in zip(critic1.parameters(), critic_target1.parameters()):
                target_param.data.copy_(target_param.data * rho + param.data * (1. - rho))

            for param, target_param in zip(critic2.parameters(), critic_target2.parameters()):
                target_param.data.copy_(target_param.data * rho + param.data * (1. - rho))

        #return actor_loss.detach().item(), critic_loss.detach().item()

def get_action(env, actor, state, noise_scale=0):

    # forward pass
    action = actor.forward(torch.FloatTensor(state)).detach().numpy()

    # add noise
    action += noise_scale * np.random.randn(env.action_space_dim)

    # clipp such that it lies in the valid action range
    action = np.clip(action, env.action_space_low, env.action_space_high)
    return action


def td3_episodic(env, gamma=0.99, d_hidden_layer=32, n_layers=3,
                 n_episodes=100, n_steps_episode_lim=1000,
                 start_steps=0, update_after=5000, update_every=100, policy_delay=50, noise_scale=0,
                 test_freq_episodes=100, backup_freq_episodes=None,
                 replay_size=50000, batch_size=512, lr_actor=1e-4, lr_critic=1e-4, test_batch_size=1000,
                 rho=0.95, seed=None,
                 value_function_hjb=None, control_hjb=None, load=False, plot=False):

    # get dir path
    rel_dir_path = get_ddpg_dir_path(
        env,
        agent='td3-episodic',
        d_hidden_layer=d_hidden_layer,
        noise_scale=noise_scale,
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
    actor = DeterministicPolicy(state_dim=d_state_space, action_dim=d_action_space,
                                hidden_sizes=actor_hidden_sizes, activation=nn.Tanh)
    actor_target = deepcopy(actor)

    # initialize critic representations
    critic_hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    critic1 = QValueFunction(state_dim=d_state_space, action_dim=d_action_space,
                            hidden_sizes=critic_hidden_sizes, activation=nn.Tanh)
    critic_target1 = deepcopy(critic1)
    critic2 = QValueFunction(state_dim=d_state_space, action_dim=d_action_space,
                            hidden_sizes=critic_hidden_sizes, activation=nn.Tanh)
    critic_target2 = deepcopy(critic2)

    # set optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_params = list(critic1.parameters()) + list(critic2.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=lr_critic)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=d_state_space, action_dim=d_action_space,
                                 size=replay_size)

    # initialize figures if plot:
    if plot and env.d == 1:
        lines_actor_critic = initialize_1d_figures(env, actor, critic1, value_function_hjb, control_hjb)
        tuple_fig_replay = initialize_replay_buffer_1d_figure(env, replay_buffer)
        lines_returns = initialize_return_and_time_steps_figures(env, n_episodes)
    elif plot and env.d == 2:
        Q_policy = initialize_2d_figures(env, actor, control_hjb)
        lines_returns = initialize_return_and_time_steps_figures(env, n_episodes)

    # save algorithm parameters
    data = {
        'gamma' : gamma,
        'batch_size' : batch_size,
        'lr_actor' : lr_actor,
        'lr_critic' : lr_critic,
        'n_episodes': n_episodes,
        'seed': seed,
        'replay_size': replay_size,
        'update_after': update_after,
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

    # define list to store results
    returns = np.empty(n_episodes)
    returns.fill(np.nan)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    time_steps.fill(np.nan)

    # preallocate lists to store test results
    test_mean_returns = np.empty((0), dtype=np.float32)
    test_var_returns = np.empty((0), dtype=np.float32)
    test_mean_lengths = np.empty((0), dtype=np.float32)
    test_policy_l2_errors = np.empty((0), dtype=np.float32)

    # get initial state
    state_init = env.state_init.copy()

    # total number of time steps
    k_total = 0

    # sample trajectories
    for ep in range(n_episodes):
        print(ep)

        # initialization
        state = env.reset()

        # reset trajectory return
        ep_return = 0

        # terminal state flag
        complete = False

        # sample trajectory
        for k in np.arange(n_steps_episode_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # sample action

            # sample action randomly
            if k_total < start_steps:
                action = env.sample_action(batch_size=1)

            # get action following the actor
            else:
                action = get_action(env, actor, state, noise_scale)

            # env step
            next_state, r, complete, _ = env.step(state, action)
            #print(k, state, action, r, next_state)

            # store tuple
            replay_buffer.store(state, action, r, next_state, complete)

            # time to update
            if k_total >= update_after and (k_total + 1) % update_every == 0:

                for l in range(update_every):

                    # sample minibatch of transition uniformlly from the replay buffer
                    batch = replay_buffer.sample_batch(batch_size)

                    # update actor and critic parameters
                    update_parameters(
                        actor, actor_target, actor_optimizer,
                        critic1, critic_target1, critic2, critic_target2, critic_optimizer,
                        batch, gamma, policy_delay, l,
                    )

            # save action and reward
            ep_return += (gamma**k) * r

            # update state
            state = next_state

            # update total steps counter
            k_total += 1

        # save episode
        returns[ep] = ep_return
        time_steps[ep] = k

        # logs
        if (ep + 1) % test_freq_episodes == 0:

            # test model
            test_mean_ret, test_var_ret, test_mean_len, test_policy_l2_error \
                    = test_policy_vectorized(env, actor, batch_size=test_batch_size,
                                             control_hjb=control_hjb)
            test_mean_returns = np.append(test_mean_returns, test_mean_ret)
            test_var_returns = np.append(test_var_returns, test_var_ret)
            test_mean_lengths = np.append(test_mean_lengths, test_mean_len)
            test_policy_l2_errors = np.append(test_policy_l2_errors, test_policy_l2_error)

            msg = 'ep: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
                  'test mean time steps: {:2.2f}, test u l2 error: {:.2e}'.format(
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
            data['test_batch_size'] = test_batch_size
            data['test_mean_returns'] = test_mean_returns
            data['test_var_returns'] = test_var_returns
            data['test_mean_lengths'] = test_mean_lengths
            data['test_policy_l2_errors'] = test_policy_l2_errors

            save_data(data, rel_dir_path)

        # update plots
        if plot and (ep + 1) % 1 == 0:
            if env.d == 1:
                update_1d_figures(env, actor, critic1, lines_actor_critic)
                update_replay_buffer_1d_figure(env, replay_buffer, tuple_fig_replay)
                update_return_and_time_steps_figures(env, returns[:ep], time_steps[:ep], lines_returns)

            elif env.d == 2:
                update_2d_figures(env, actor, Q_policy)
                update_return_and_time_steps_figures(env, returns[:ep], time_steps[:ep], lines_returns)

    data['returns'] = returns
    data['time_steps'] = time_steps
    data['test_batch_size'] = test_batch_size
    data['test_mean_returns'] = test_mean_returns
    data['test_var_returns'] = test_var_returns
    data['test_mean_lengths'] = test_mean_lengths
    data['test_policy_l2_errors'] = test_policy_l2_errors
    save_data(data, rel_dir_path)
    return data

def initialize_1d_figures(env, actor, critic1, value_function_hjb, control_hjb):
    q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic1)
    v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic1)
    lines = initialize_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                            a_table, policy_actor, policy_critic,
                                            value_function_hjb, control_hjb)
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

