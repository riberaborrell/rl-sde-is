from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import *
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
        return torch.squeeze(q, -1)

def pre_train_critic(env, critic):

    # optimizer
    optimizer = optim.Adam(critic.parameters(), lr=1e-4)

    # batch size
    batch_size = 10**3

    # number of iterations
    n_iterations = 10**4

    def sample_data_points():
        #states = torch.normal(mean=0, std=1, size=(batch_size, 1))
        #actions = torch.normal(mean=0, std=5, size=(batch_size, 1))
        states = torch.distributions.uniform.Uniform(-2, 2).sample([batch_size, 1])
        actions = torch.distributions.uniform.Uniform(-10, 10).sample([batch_size, 1])

        idx_ts = torch.where(states > 1)[0]
        idx_not_ts = torch.where(states < 1)[0]

        # targets
        q_values_target = torch.empty((batch_size, 1))
        q_values_target[idx_ts] = 0.
        q_values_target[idx_not_ts] = -1.

        return states, actions, q_values_target


    # train
    for i in range(n_iterations):

        # sample data
        states, actions, q_values_target = sample_data_points()

        # compute q values
        q_values = critic.forward(states, actions)

        # compute mse loss
        loss = ((q_values - q_values_target)**2).mean()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Critic pre-trained to have null in the target set and negative values elsewhere')

def pre_train_actor(env, actor):

    optimizer = optim.Adam(actor.parameters(), lr=0.001)

    # batch size
    batch_size = 10**2

    # number of iterations
    n_iterations = 10**2

    def sample_data_points():
        states = (env.state_space_low - env.state_space_high) * torch.rand(batch_size, 1) + env.state_space_high
        return states

    # targets
    actions_target = torch.zeros((batch_size, 1))

    # train
    for i in range(n_iterations):

        if i % 10 == 0:

            # sample points
            states = sample_data_points()

        # compute actions
        actions = actor.forward(states)

        # compute mse loss
        loss = ((actions - actions_target)**2).mean()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Actor pre-trained to have null actions')

def get_action(env, actor, state, noise_scale=0):

    # forward pass
    action = actor.forward(torch.FloatTensor(state)).detach().numpy()

    # add noise
    action += noise_scale * np.random.randn(env.action_space_dim)

    # clipp such that it lies in the valid action range
    action = np.clip(action, env.action_space_low, env.action_space_high)
    return action

def update_parameters(actor, actor_target, actor_optimizer, critic, critic_target,
                      critic_optimizer, batch, gamma, rho=0.95):

    # unpack tuples in batch
    states = torch.tensor(batch['state'])
    next_states = torch.tensor(batch['next_state'])
    actions = torch.tensor(batch['act'])
    rewards = torch.tensor(batch['rew'])
    done = torch.tensor(batch['done'])

    # get batch size
    batch_size = states.shape[0]

    # 1) run 1 gradient descent step for Q (critic)

    # reset critic gradients
    critic_optimizer.zero_grad()

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
    critic_loss = ((q_vals - target)**2).mean()

    # update critic network
    critic_loss.backward()
    critic_optimizer.step()

    # 2) run 1 gradient descent step for mu (actor)

    # reset actor gradients
    actor_optimizer.zero_grad()

    # freeze Q-network to save computational effort 
    for param in critic.parameters():
            param.requires_grad = False

    # actor loss
    actor_loss = - critic.forward(states, actor.forward(states)).mean()

    # update actor network
    actor_loss.backward()
    actor_optimizer.step()

    # unfreeze Q-network to save computational effort 
    for param in critic.parameters():
            param.requires_grad = True

    # update actor and critic target networks "softly”
    with torch.no_grad():
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(target_param.data * rho + param.data * (1. - rho))

        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(target_param.data * rho + param.data * (1. - rho))

    return actor_loss.detach().item(), critic_loss.detach().item()


def ddpg_episodic(env, gamma=0.99, d_hidden_layer=256, n_layers=3,
         n_episodes=100, n_steps_episode_lim=1000,
         start_steps=0, update_after=5000, update_every=100,
         replay_size=50000, batch_size=512, lr_actor=1e-4, lr_critic=1e-4,
         test_freq_episodes=100, test_batch_size=1000, backup_freq_episodes=None,
         seed=None, value_function_hjb=None, control_hjb=None, load=False, plot=False):

    # get dir path
    rel_dir_path = get_ddpg_dir_path(
        env,
        agent='ddpg-episodic',
        d_hidden_layer=d_hidden_layer,
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
    critic = QValueFunction(state_dim=d_state_space, action_dim=d_action_space,
                            hidden_sizes=critic_hidden_sizes, activation=nn.Tanh)
    critic_target = deepcopy(critic)

    # set optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    # pre-train actor and critic
    #pre_train_critic(env, critic)
    #pre_train_actor(env, actor)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=d_state_space, action_dim=d_action_space,
                                 size=replay_size)

    # initialize figures
    if plot:
        q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic)
        v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)
        lines = initialize_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                                a_table, policy_actor, policy_critic,
                                                value_function_hjb, control_hjb)
        tuple_fig_replay = initialize_replay_buffer_1d_figure(env, replay_buffer)
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
        'critic': critic,
        'rel_dir_path': rel_dir_path,
    }
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-epi{}'.format(0))
    save_model(critic, rel_dir_path, 'critic_n-epi{}'.format(0))

    # define list to store results
    returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)

    # preallocate lists to store test results
    test_mean_returns = np.empty((0), dtype=np.float32)
    test_var_returns = np.empty((0), dtype=np.float32)
    test_mean_lengths = np.empty((0), dtype=np.float32)
    test_policy_l2_errors = np.empty((0), dtype=np.float32)

    # get initial state
    state_init = env.state_init.copy()

    # sample trajectories
    for ep in range(n_episodes):
        print(ep)

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
            if k < start_steps:
                action = env.sample_action(batch_size=1)

            # get action following the actor
            else:
                action = get_action(env, actor, state, noise_scale=2.)

            # env step
            next_state, r, done, _ = env.step(state, action)

            # store tuple
            replay_buffer.store(state, action, r, next_state, done)

            # if buffer is full enough
            if replay_buffer.size > update_after and (k + 1) % update_every == 0:
                for _ in range(update_every):

                    # sample minibatch of transition uniformlly from the replay buffer
                    batch = replay_buffer.sample_batch(batch_size)

                    # update actor and critic parameters
                    actor_loss, critic_loss = update_parameters(
                        actor, actor_target, actor_optimizer,
                        critic, critic_target, critic_optimizer,
                        batch, gamma,
                    )

            # save action and reward
            ep_return += (gamma**k) * r
            #print(ep_return)

            # update state
            state = next_state

        # save episode
        returns[ep] = ep_return
        time_steps[ep] = k

        # test actor
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
            save_model(critic, rel_dir_path, 'critic_n-epi{}'.format(ep + 1))

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
            q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic)
            v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)
            update_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                    a_table, policy_actor, policy_critic, lines)
            update_replay_buffer_1d_figure(env, replay_buffer, tuple_fig_replay)

    data['returns'] = returns
    data['time_steps'] = time_steps
    data['test_batch_size'] = test_batch_size
    data['test_mean_returns'] = test_mean_returns
    data['test_var_returns'] = test_var_returns
    data['test_mean_lengths'] = test_mean_lengths
    data['test_policy_l2_errors'] = test_policy_l2_errors
    save_data(data, rel_dir_path)
    return data

def ddpg_continuing(env, gamma=0.99, d_hidden_layer=32, n_layers=3,
         n_total_steps=100000, n_steps_episode_lim=500,
         start_steps=0, update_after=5000, update_every=100,
         replay_size=50000, batch_size=512, lr_actor=1e-4, lr_critic=1e-4,
         test_freq_steps=10000, test_batch_size=1000, backup_freq_steps=None,
         seed=None, value_function_hjb=None, control_hjb=None, load=False, plot=False):

    # get dir path
    rel_dir_path = get_ddpg_dir_path(
        env,
        agent='ddpg-continuing',
        d_hidden_layer=d_hidden_layer,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        n_total_steps=n_total_steps,
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
    critic = QValueFunction(state_dim=d_state_space, action_dim=d_action_space,
                            hidden_sizes=critic_hidden_sizes, activation=nn.Tanh)
    critic_target = deepcopy(critic)

    # set optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=d_state_space, action_dim=d_action_space,
                                 size=replay_size)
    # initialize figures
    if plot:
        q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic)
        v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)
        lines = initialize_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                                a_table, policy_actor, policy_critic,
                                                value_function_hjb, control_hjb)
        tuple_fig_replay = initialize_replay_buffer_1d_figure(env, replay_buffer)

    # save algorithm parameters
    data = {
        'gamma' : gamma,
        'batch_size' : batch_size,
        'lr_actor' : lr_actor,
        'lr_critic' : lr_critic,
        'n_total_steps': n_total_steps,
        'seed': seed,
        'replay_size': replay_size,
        'update_after': update_after,
        'actor': actor,
        'critic': critic,
        'rel_dir_path': rel_dir_path,
    }
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-epi{}'.format(0))
    save_model(critic, rel_dir_path, 'critic_n-epi{}'.format(0))

    # define list to store episodic results
    returns = np.empty((0), dtype=np.float32)
    time_steps = np.empty((0), dtype=np.float32)

    # preallocate lists to store test results
    test_mean_returns = np.empty((0), dtype=np.float32)
    test_var_returns = np.empty((0), dtype=np.float32)
    test_mean_lengths = np.empty((0), dtype=np.float32)
    test_policy_l2_errors = np.empty((0), dtype=np.float32)

    # reset episode
    state, rew, done, ep_ret, ep_len, ep = env.reset(), 0, False, 0, 0, 0

    for k in range(n_total_steps):

        # sample action

        # sample action randomly
        if k < start_steps:
            action = env.sample_action(batch_size=1)

        # get action following the actor
        else:
            action = get_action(env, actor, state, noise_scale=0)

        # step dynamics forward
        next_state, rew, done, _ = env.step(state, action)

        # store tuple
        replay_buffer.store(state, action, rew, next_state, done)

        # update state
        state = next_state

        # update episode return and length
        ep_ret += rew
        ep_len += 1

        # update step when buffer is full enough
        if k >= update_after and (k + 1) % update_every == 0:
            for _ in range(update_every):

                # sample minibatch of transition uniformly from the replay buffer
                batch = replay_buffer.sample_batch(batch_size)

                # update actor and critic parameters
                actor_loss, critic_loss = update_parameters(
                #update_parameters(
                    actor, actor_target, actor_optimizer,
                    critic, critic_target, critic_optimizer,
                    batch, gamma,
                )

        # if trajectory is complete
        if done:
            msg = 'total k: {:3d}, ep: {:3d}, return {:2.2f}, time steps: {:2.2f}'.format(
                    k,
                    ep,
                    ep_ret.item(),
                    ep_len,
                )
            print(msg)
            returns = np.append(returns, ep_ret)
            time_steps = np.append(time_steps, ep_len)

            # reset episode 
            state, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            ep += 1

        # when epoch is finish evaluate the agent
        if (k + 1) % test_freq_steps == 0:

            # test model
            test_mean_ret, test_var_ret, test_mean_len, test_policy_l2_error \
                    = test_policy_vectorized(env, actor, batch_size=test_batch_size,
                                             control_hjb=control_hjb)
            test_mean_returns = np.append(test_mean_returns, test_mean_ret)
            test_var_returns = np.append(test_var_returns, test_var_ret)
            test_mean_lengths = np.append(test_mean_lengths, test_mean_len)

            msg = 'total k: {:3d}, test mean return: {:2.2f}, test var return: {:.2e},' \
                  'test mean time steps: {:2.2f} '.format(
                k,
                test_mean_ret,
                test_var_ret,
                test_mean_len,
            )
            print(msg)

        # backup models and results
        if backup_freq_steps is not None and (k + 1) % backup_freq_steps == 0:

            # save actor and critic models
            save_model(actor, rel_dir_path, 'actor_n-step{}'.format(k + 1))
            save_model(critic, rel_dir_path, 'critic_n-step{}'.format(k + 1))

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
        if plot and (k + 1) % 100 == 0:
            q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic)
            v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)
            update_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                        a_table, policy_actor, policy_critic, lines)
            update_replay_buffer_1d_figure(env, replay_buffer, tuple_fig_replay)


    data['returns'] = returns
    data['time_steps'] = time_steps
    data['test_batch_size'] = test_batch_size
    data['test_mean_returns'] = test_mean_returns
    data['test_var_returns'] = test_var_returns
    data['test_mean_lengths'] = test_mean_lengths
    data['test_policy_l2_errors'] = test_policy_l2_errors
    save_data(data, rel_dir_path)
    return data

def load_backup_models(data, ep=0):
    actor = data['actor']
    critic = data['critic']
    rel_dir_path = data['rel_dir_path']
    try:
        load_model(actor, rel_dir_path, file_name='actor_n-epi{}'.format(ep))
        load_model(critic, rel_dir_path, file_name='critic_n-epi{}'.format(ep))
    except FileNotFoundError as e:
        print('there is no backup after episode {:d}'.format(ep))

