from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl_sde_is.dpg.dpg_utils import DeterministicPolicy, QValueFunction
from rl_sde_is.dpg.replay_buffers import ReplayBuffer
from rl_sde_is.utils.approximate_methods import *
from rl_sde_is.utils.path import get_ddpg_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import *


def get_action(env, actor, state, noise_scale, action_limit):

    # forward pass
    action = actor.forward(torch.FloatTensor(state)).detach().numpy()

    # add noise
    action += noise_scale * np.random.randn(env.action_space.shape[0])

    # clipp such that it lies in the valid action range
    action = np.clip(action, -action_limit, action_limit)
    return action

def update_parameters(actor, actor_target, actor_optimizer, critic, critic_target,
                      critic_optimizer, batch, gamma, polyak):

    # unpack tuples in batch
    states = torch.tensor(batch['states'])
    actions = torch.tensor(batch['actions'])
    rewards = torch.tensor(batch['rewards'])
    next_states = torch.tensor(batch['next_states'])
    done = torch.tensor(batch['done'])

    # get batch size
    batch_size = states.shape[0]

    # 1) run 1 gradient descent step for Q (critic)

    # q value for the given pairs of states and actions (forward pass of the critic network)
    q_vals = critic.forward(states, actions)

    with torch.no_grad():

        # q value for the corresponding next pair of states and actions (using target networks)
        next_actions = actor_target.forward(next_states).detach()
        q_vals_next = critic_target.forward(next_states, next_actions)

        # compute target (using target networks)
        d = torch.where(done, 1., 0.)
        target = rewards + gamma * (1. - d) * q_vals_next

    # critic loss
    critic_loss = (q_vals - target).pow(2).mean()

    # update critic network
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # 2) run 1 gradient descent step for mu (actor)

    # freeze Q-network to save computational effort 
    for param in critic.parameters():
        param.requires_grad = False

    # actor loss
    actor_loss = - critic.forward(states, actor.forward(states)).mean()

    # update actor network
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # unfreeze Q-network to save computational effort 
    for param in critic.parameters():
        param.requires_grad = True

    # update actor and critic target networks "softly”
    with torch.no_grad():
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1. - polyak))

        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1. - polyak))

    return actor_loss.detach().item(), critic_loss.detach().item()


def ddpg_episodic(env, gamma=1., n_layers=2, d_hidden_layer=32,
         n_episodes=100, n_steps_lim=1000, learning_starts=1000, replay_size=50000,
         batch_size=512, lr_actor=1e-4, lr_critic=1e-4, seed=None,
         update_freq=100, polyak=0.95, expl_noise=0., action_limit=None,
         backup_freq=None, live_plot_freq=None,
         value_function_opt=None, policy_opt=None, load=False):

    # get dir path
    dir_path = get_ddpg_dir_path(
        env,
        agent='ddpg-episodic',
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
        expl_noise=expl_noise,
        action_limit=action_limit,
        polyak=polyak,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
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

    # initialize actor representations
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    actor = DeterministicPolicy(state_dim=env.d, action_dim=env.d,
                                hidden_sizes=hidden_sizes, activation=nn.Tanh())
    actor_target = deepcopy(actor)

    # initialize critic representations
    critic = QValueFunction(state_dim=env.d, action_dim=env.d,
                            hidden_sizes=hidden_sizes, activation=nn.Tanh())
    critic_target = deepcopy(critic)

    # set optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=env.d, action_dim=env.d, size=replay_size)

    # save algorithm parameters
    data = {
        'gamma' : gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'n_episodes': n_episodes,
        'n_steps_lim': n_steps_lim,
        'expl_noise': expl_noise,
        'action_limit': action_limit,
        'replay_size': replay_size,
        'batch_size' : batch_size,
        'lr_actor' : lr_actor,
        'lr_critic' : lr_critic,
        'seed': seed,
        'learning_starts': learning_starts,
        'update_freq': update_freq,
        'actor': actor,
        'critic': critic,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)


    # save models initial parameters
    save_model(actor, dir_path, 'actor_n-epi{}'.format(0))
    save_model(critic, dir_path, 'critic_n-epi{}'.format(0))

    # define list to store results
    returns = np.full(n_episodes, np.nan, dtype=np.float32)
    time_steps = np.full(n_episodes, np.nan, dtype=np.int32)
    cts = np.full(n_episodes, np.nan, dtype=np.float32)

    # total number of time steps
    k_total = 0

    # initialize figures if plot:
    if live_plot_freq and env.d == 1:
        lines_actor_critic = initialize_1d_figures(env, actor, critic, value_function_opt, policy_opt)
        tuple_fig_replay = initialize_replay_buffer_1d_figure(env, replay_buffer)
        lines_returns = initialize_return_and_time_steps_figures(env, n_episodes)
    elif live_plot_freq and env.d == 2:
        Q_policy = initialize_2d_figures(env, actor, policy_opt)
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
                #action = env.sample_action(batch_size=1)
                action = env.action_space.sample()

            # get action following the actor
            else:
                action = get_action(env, actor, state, expl_noise, action_limit)

            # env step
            next_state, r, done, _, info = env.step(action)

            # store tuple
            replay_buffer.store(state, action, r, next_state, done)

            # time to update
            if k_total >= learning_starts and (k_total + 1) % update_freq == 0:
                for _ in range(update_freq):

                    # sample minibatch of transition uniformlly from the replay buffer
                    batch = replay_buffer.sample_batch(batch_size)

                    # update actor and critic parameters
                    actor_loss, critic_loss = update_parameters(
                        actor, actor_target, actor_optimizer,
                        critic, critic_target, critic_optimizer,
                        batch, gamma, polyak,
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

        msg = 'ep.: {:2d}, return: {:.3e}, time steps: {:.3e}, ct: {:.3f}'.format(
            ep,
            returns[ep],
            time_steps[ep],
            cts[ep],
        )
        print(msg)

        # backup models and results
        if backup_freq is not None and (ep + 1) % backup_freq == 0:

            # save actor and critic models
            save_model(actor, dir_path, 'actor_n-epi{}'.format(ep + 1))
            save_model(critic, dir_path, 'critic_n-epi{}'.format(ep + 1))

            # save test results
            data['returns'] = returns
            data['time_steps'] = time_steps
            data['cts'] = cts
            save_data(data, dir_path)

        # update plots
        if live_plot_freq and (ep + 1) % live_plot_freq == 0:
            if env.d == 1:
                update_1d_figures(env, actor, critic, lines_actor_critic)
                update_replay_buffer_1d_figure(env, replay_buffer, tuple_fig_replay)
                update_return_and_time_steps_figures(env, returns[:ep], time_steps[:ep], lines_returns)

            elif env.d == 2:
                update_2d_figures(env, actor, Q_policy)
                update_return_and_time_steps_figures(env, returns[:ep], time_steps[:ep], lines_returns)

    # add learning results
    data['returns'] = returns
    data['time_steps'] = time_steps
    data['cts'] = cts
    data['replay_states'] = replay_buffer.states[:replay_buffer.size]
    data['replay_actions'] = replay_buffer.actions[:replay_buffer.size]
    save_data(data, dir_path)
    return data

def initialize_1d_figures(env, actor, critic, value_function_opt, policy_opt):
    q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
    v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
    lines = initialize_actor_critic_1d_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                               a_table, policy_actor, policy_critic,
                                               value_function_opt, policy_opt)
    return lines

def update_1d_figures(env, actor, critic, lines):
    q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
    v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
    update_actor_critic_1d_figures(env, q_table, v_table_actor_critic, v_table_critic,
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
    critic = data['critic']
    dir_path = data['dir_path']
    try:
        load_model(actor, dir_path, file_name='actor_n-epi{}'.format(ep))
        load_model(critic, dir_path, file_name='critic_n-epi{}'.format(ep))
    except FileNotFoundError as e:
        print('The episode {:d} has no backup '.format(ep))
