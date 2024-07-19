import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import *
from rl_sde_is.utils.plots import *
from rl_sde_is.dpg.replay_buffers import ReplayBuffer
from rl_sde_is.utils.path import *
from rl_sde_is.utils.tests import TestPolicy
from rl_sde_is.dpg.dpg_utils import DeterministicPolicy, QValueFunction
#from rl_sde_is.dpg_utils import DeterministicPolicy, ValueFunction, AValueFunction

def initialize_actor_and_critic(env, **kwargs):
    # dimensions of the state and action space
    d_state_space = env.state_space_dim
    d_action_space = env.action_space_dim
    n_layers = kwargs['n_layers']
    d_hidden_layer = kwargs['d_hidden_layer']

    # initialize actor representations
    actor_hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    actor = DeterministicPolicy(
        state_dim=d_state_space,
        action_dim=d_action_space,
        hidden_sizes=actor_hidden_sizes,
        activation=nn.Tanh(),
    )

    # initialize critic representations
    critic_hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    critic_q = QValueFunction(
        state_dim=d_state_space,
        action_dim=d_action_space,
        hidden_sizes=critic_hidden_sizes,
        activation=nn.Tanh()
    )
    critic_q = train_critic_from_dp(env, critic_q, kwargs['value_function_opt'],
                                    kwargs['policy_opt'], load=True)
    """
    critic_v = ValueFunction(
        state_dim=d_state_space,
        hidden_sizes=critic_hidden_sizes,
        activation=nn.Tanh()
    )
    critic_a = AValueFunction(
        state_dim=d_state_space,
        action_dim=d_action_space,
        hidden_sizes=critic_hidden_sizes,
        activation=nn.Tanh()
    )
    """
    #critic_v, critic_a = train_dueling_critic_from_dp(env, critic_v, critic_a, value_function_opt, policy_opt)#, load=True)

    # set optimizers
    #actor_optimizer = optim.SGD(actor.parameters(), lr=lr_actor)
    actor_optimizer = optim.Adam(actor.parameters(), lr=kwargs['lr_actor'])

    return actor, actor_optimizer, critic_q



def update_actor_critic_naive(actor, actor_optimizer, critic, critic_optimizer, experience):

    # unpack experience
    state, action, reward, next_state, done, fht_estimate = experience

    # torchify experience
    state = torch.FloatTensor(state)
    action = torch.FloatTensor(action)
    reward = torch.tensor(reward)
    next_state = torch.FloatTensor(next_state)
    done = torch.tensor(done)

    # 1) critic loss

    # reset critic gradients
    critic_optimizer.zero_grad()

    # q value for the state-action pair
    q_val = critic.forward(state, action)

    with torch.no_grad():

        # next action following the actor
        next_action = actor.forward(next_state).detach()

        # q value for the next-state-next-action pair
        q_val_next = critic.forward(next_state, next_action)

        # target
        d = torch.where(done, 1., 0.)
        target = reward + (1. - d) * q_val_next

    critic_loss = (q_val - target)**2
    #print('q_val: {:2.3f}, target: {:2.3f}'.format(q_val, target))

    # update critic network
    critic_loss.backward()
    critic_optimizer.step()

    # freeze q-networks to save computational effort 
    for param in critic.parameters():
        param.requires_grad = False

    # 2) actor loss

    # reset actor gradients
    actor_optimizer.zero_grad()

    actor_loss = - fht_estimate * critic.forward(state, actor.forward(state))

    #update actor network
    actor_loss.backward()
    actor_optimizer.step()

    # unfreeze q-network to save computational effort 
    for param in critic.parameters():
        param.requires_grad = True

def update_actor_critic_simple(actor, actor_optimizer, critic, critic_optimizer, batch,
                               fht_estimate=1.0):

    # unpack tuples in batch
    states = torch.tensor(batch['states'])
    actions = torch.tensor(batch['actions'])
    rewards = torch.tensor(batch['rewards'])
    next_states = torch.tensor(batch['next_states'])
    done = torch.tensor(batch['done'])

    # get batch size
    batch_size = states.shape[0]

    # 1) critic loss

    # reset critic gradients
    critic_optimizer.zero_grad()

    # q value for the given pairs of states and actions
    q_vals = critic.forward(states, actions)

    with torch.no_grad():

        # next actions following the actor
        next_actions = actor.forward(next_states).detach()

        # q value for the next-state-next-action pair
        q_vals_next = critic.forward(next_states, next_actions)

        d = torch.where(done, 1., 0.)
        targets = rewards + (1. - d) * q_vals_next

    # critic loss
    critic_loss = ((q_vals - targets)**2).mean()

    # update critic network
    critic_loss.backward()
    critic_optimizer.step()

    # freeze q-networks to save computational effort 
    for param in critic.parameters():
        param.requires_grad = False

    # reset actor gradients
    actor_optimizer.zero_grad()

    # 2) actor loss
    actor_loss = - fht_estimate * critic.forward(states, actor.forward(states)).mean()

    #update actor network
    actor_loss.backward()
    actor_optimizer.step()

    # unfreeze q-network to save computational effort 
    for param in critic.parameters():
        param.requires_grad = True

#def update_parameters(actor, actor_optimizer, critic, batch):
def update_parameters(actor, actor_optimizer, critic, states):

    # unpack tuples in batch
    #states = torch.tensor(batch['states'])
    states = torch.tensor(states)

    # get batch size
    batch_size = states.shape[0]

    # freeze q-networks to save computational effort 
    for param in critic.parameters():
        param.requires_grad = False

    # actor loss
    actor_loss = - critic.forward(states, actor.forward(states)).mean()

    # reset actor gradients
    actor_optimizer.zero_grad()

    #update actor network
    actor_loss.backward()
    actor_optimizer.step()

    # unfreeze q-network to save computational effort 
    for param in critic.parameters():
        param.requires_grad = True


def dpg_actor_critic_naive(env, **kwargs):

    # get dir path
    rel_dir_path = get_dpg_dir_path(env, agent='dpg-naive', **kwargs)

    # load results
    if kwargs['load']:
        return load_data(rel_dir_path)

    # set seed
    seed = kwargs['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initialize models and optimizers
    hidden_sizes = [kwargs['d_hidden_layer'] for i in range(kwargs['n_layers'] -1)]
    actor = DeterministicPolicy(state_dim=env.d, action_dim=env.d,
                                hidden_sizes=hidden_sizes, activation=nn.Tanh())

    # initialize critic representations
    critic = QValueFunction(state_dim=env.d, action_dim=env.d,
                            hidden_sizes=hidden_sizes, activation=nn.Tanh())

    actor_optimizer = optim.Adam(actor.parameters(), lr=kwargs['lr_actor'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=kwargs['lr_critic'])

    # save algorithm relevant parameters
    data = {
        key: value for key, value in kwargs.items() if key in [
            'd_hidden_layer',
            'n_layers',
            'n_steps_episode_lim',
            'lr_actor',
            'lr_critic',
            'n_episodes',
            'seed',
    ]}
    data['actor'] = actor
    data['critic'] = critic
    data['rel_dir_path'] = rel_dir_path
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-ep{}'.format(0))
    save_model(critic, rel_dir_path, 'critic_n-ep{}'.format(0))

    # ct per iterations
    n_episodes = kwargs['n_episodes']
    n_steps_lim = kwargs['n_steps_lim']
    cts = np.empty(n_episodes, dtype=np.float32)
    cts.fill(np.nan)

    # initialize figures if plot:
    live_plot = kwargs['live_plot']
    if live_plot:
        q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
        v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
        lines = initialize_actor_critic_1d_figures(
            env, q_table, v_table_actor_critic, v_table_critic, a_table,
            policy_actor, policy_critic, kwargs['value_function_opt'], kwargs['policy_opt'])

    for ep in range(n_episodes):

        # initialization
        state, _ = env.reset()

        # terminal state flag
        done = False

        # sample trajectories
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # sample action
            action = actor.forward(torch.FloatTensor(state)).detach().numpy()

            # env step
            next_state, r, done, _, info = env.step(action)

            # store tuple
            #fht_estimate = env.lengths[ep-1] if ep > 0 else 1.
            fht_estimate = 1.
            experience = (state, action, r, next_state, done, fht_estimate)

            # update actor parameters
            update_actor_critic_naive(actor, actor_optimizer, critic, critic_optimizer, experience)

        print('ep: {:3d}'.format(ep+1))

        # backup models
        backup_freq_episodes = kwargs['backup_freq_episodes']
        if backup_freq_episodes is not None and (ep + 1) % backup_freq_episodes == 0:

            # save actor and critic models
            save_model(actor, rel_dir_path, 'actor_n-ep{}'.format(ep + 1))
            save_model(critic, rel_dir_path, 'critic_n-ep{}'.format(ep + 1))

        # update plots
        if live_plot and (ep + 1) % 1 == 0:
            q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
            v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
            update_actor_critic_1d_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                           a_table, policy_actor, policy_critic, lines)


    # add learning results
    data['cts'] = cts

    save_data(data, rel_dir_path)
    return data

def dpg_actor_critic_batch(env, **kwargs):

    # get dir path
    rel_dir_path = get_dpg_dir_path(env, agent='dpg-batch', **kwargs,)

    # load results
    if kwargs['load']:
        return load_data(rel_dir_path)

    # set seed
    seed = kwargs['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initialize models and optimizers
    hidden_sizes = [kwargs['d_hidden_layer'] for i in range(kwargs['n_layers'] -1)]
    actor = DeterministicPolicy(state_dim=env.d, action_dim=env.d,
                                hidden_sizes=hidden_sizes, activation=nn.Tanh())

    # initialize critic representations
    critic = QValueFunction(state_dim=env.d, action_dim=env.d,
                            hidden_sizes=hidden_sizes, activation=nn.Tanh())

    actor_optimizer = optim.Adam(actor.parameters(), lr=kwargs['lr_actor'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=kwargs['lr_critic'])

    # initialize replay buffer
    batch_size = kwargs['batch_size']
    n_episodes = kwargs['n_episodes']
    replay_buffer = ReplayBuffer(size=int(1e5), state_dim=env.d, action_dim=env.d)

    # save algorithm relevant parameters
    data = {
        key: value for key, value in kwargs.items() if key in [
            'd_hidden_layer',
            'n_layers',
            'n_steps_lim',
            'lr_actor',
            'lr_critic',
            'n_iterations',
            'seed',
    ]}
    data['actor'] = actor
    data['critic'] = critic
    data['rel_dir_path'] = rel_dir_path
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-ep{}'.format(0))
    save_model(critic, rel_dir_path, 'critic_n-ep{}'.format(0))

    # ct per iterations
    n_iterations = kwargs['n_iterations']
    n_steps_lim = kwargs['n_steps_lim']

    # initialize figures if plot:
    live_plot = kwargs['live_plot']
    if live_plot:
        q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
        v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
        lines = initialize_actor_critic_1d_figures(
            env, q_table, v_table_actor_critic, v_table_critic, a_table,
            policy_actor, policy_critic, kwargs['value_function_opt'], kwargs['policy_opt'])

    for it in range(n_iterations):

        # sample trajectories
        env.reset_statistics()
        sample_trajectories_buffer(env, actor, replay_buffer, n_episodes, n_steps_lim)
        fht_estimate = env.lengths.mean()

        for _ in range(100):

            # sample minibatch of transition uniformlly from the replay buffer
            batch = replay_buffer.sample_batch(batch_size)

            # update actor and critic parameters
            update_actor_critic_simple(actor, actor_optimizer, critic, critic_optimizer,
                                       batch, fht_estimate)

        # reset replay buffer
        replay_buffer.reset()

        print('it: {:3d}'.format(it+1))

        # update plots
        if live_plot and (it + 1) % 1 == 0:
            q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
            v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
            update_actor_critic_1d_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                           a_table, policy_actor, policy_critic, lines)


    save_data(data, rel_dir_path)
    return data

def dpg_batch(env, **kwargs):

    # get dir path
    rel_dir_path = get_dpg_dir_path(env, agent='dpg-batch', **kwargs,)

    # load results
    load = kwargs['load']
    if load:
        return load_data(rel_dir_path)

    # set seed
    seed = kwargs['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


    # initialize models and optimizer
    actor, actor_optimizer, critic = initialize_actor_and_critic(env, **kwargs)

    # initialize replay buffer
    batch_size = kwargs['batch_size']
    replay_buffer = ReplayBuffer(
        size=int(1e5), state_dim=env.state_space_dim, action_dim=env.action_space_dim,
    )

    # save algorithm relevant parameters
    data = {
        key: value for key, value in kwargs.items() if key in [
            'gamma',
            'd_hidden_layer',
            'n_layers',
            'n_steps_episode_lim',
            'batch_size',
            'lr_actor',
            'n_iterations',
            'seed',
    ]}
    data['actor'] = actor
    data['rel_dir_path'] = rel_dir_path
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-it{}'.format(0))

    # ct per iterations
    n_iterations = kwargs['n_iterations']
    cts = np.empty(n_iterations, dtype=np.float32)
    cts.fill(np.nan)

    # test policy
    test = kwargs['test']
    policy_opt = kwargs['policy_opt']
    if test:
        test_batch_size = kwargs['test_batch_size']
        test_freq_iterations = kwargs['test_freq_iterations']
        test_policy = TestPolicy(test_batch_size, test_freq_iterations, freq_type='it')
        test_policy.test_actor_model(env, actor, policy_opt)

    # get initial state
    state_init = env.state_init.copy()

    # initialize figures if plot:
    live_plot = kwargs['live_plot']
    if live_plot:
        policy = compute_table_det_policy_1d(env, actor)
        line_actor = initialize_det_policy_1d_figure(env, policy, policy_opt)

    for it in range(n_iterations):

        # start timer
        ct_initial = time.time()

        # sample trajectories
        sample_trajectories_buffer_vectorized(env, actor, replay_buffer,
                                              batch_size, kwargs['n_steps_episode_lim'])

        # sample minibatch of transition uniformlly from the replay buffer
        batch = replay_buffer.sample_batch(batch_size)

        # update actor parameters
        update_parameters(actor, actor_optimizer, critic, batch['states'])

        # reset replay buffer
        replay_buffer.reset()

        # end timer
        ct_final = time.time()

        # save episode
        cts[ep] = ct_final - ct_initial

        #print('it: {:3d}, fht: {:2.2f}'.format(it+1, fht))

        # update plots
        if live_plot and (ep + 1) % 1 == 0:
            policy = compute_table_det_policy_1d(env, actor)
            update_det_policy_1d_figure(env, policy, line_actor)

    # add learning results
    data['cts'] = cts

    save_data(data, rel_dir_path)
    return data


def dpg_replay(env, **kwargs):

    # get dir path
    rel_dir_path = get_dpg_dir_path(env, agent='dpg-replay', **kwargs)

    # load results
    load = kwargs['load']
    if load:
        return load_data(rel_dir_path)

    # set seed
    seed = kwargs['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


        # update actor parameters
        #update_actor_critic_naive(actor, actor_optimizer, critic, batch['states'])

        # end timer
        #ct_final = time.time()

        # save episode
        cts[ep] = ct_final - ct_initial

        #print('it: {:3d}, fht: {:2.2f}'.format(it+1, fht))

        # update plots
        if live_plot and (it + 1) % 1 == 0:
            policy = compute_table_det_policy_1d(env, actor)
            update_det_policy_1d_figure(env, policy, line_actor)

    # add learning results
    data['cts'] = cts

    save_data(data, rel_dir_path)
    return data


def dpg_batch(env, **kwargs):

    # get dir path
    rel_dir_path = get_dpg_dir_path(env, agent='dpg-batch', **kwargs,)

    # load results
    load = kwargs['load']
    if load:
        return load_data(rel_dir_path)

    # set seed
    seed = kwargs['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


    # initialize models and optimizer
    actor, actor_optimizer, critic = initialize_actor_and_critic(env, **kwargs)

    # initialize replay buffer
    batch_size = kwargs['batch_size']
    replay_buffer = ReplayBuffer(
        size=int(1e5), state_dim=env.state_space_dim, action_dim=env.action_space_dim,
    )

    # save algorithm relevant parameters
    data = {
        key: value for key, value in kwargs.items() if key in [
            'gamma',
            'd_hidden_layer',
            'n_layers',
            'n_steps_episode_lim',
            'batch_size',
            'lr_actor',
            'n_iterations',
            'seed',
    ]}
    data['actor'] = actor
    data['rel_dir_path'] = rel_dir_path
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-it{}'.format(0))

    # ct per iterations
    n_iterations = kwargs['n_iterations']
    cts = np.empty(n_iterations, dtype=np.float32)
    cts.fill(np.nan)

    # test policy
    test = kwargs['test']
    policy_opt = kwargs['policy_opt']
    if test:
        test_batch_size = kwargs['test_batch_size']
        test_freq_iterations = kwargs['test_freq_iterations']
        test_policy = TestPolicy(test_batch_size, test_freq_iterations, freq_type='it')
        test_policy.test_actor_model(env, actor, policy_opt)

    # get initial state
    state_init = env.state_init.copy()

    # initialize figures if plot:
    live_plot = kwargs['live_plot']
    if live_plot:
        policy = compute_table_det_policy_1d(env, actor)
        line_actor = initialize_det_policy_1d_figure(env, policy, policy_opt)

    for it in range(n_iterations):

        # start timer
        ct_initial = time.time()

        # sample trajectories
        sample_trajectories_buffer_vectorized(env, actor, replay_buffer,
                                              batch_size, kwargs['n_steps_episode_lim'])

        # sample minibatch of transition uniformlly from the replay buffer
        batch = replay_buffer.sample_batch(batch_size)

        # update actor parameters
        update_parameters(actor, actor_optimizer, critic, batch['states'])

        # reset replay buffer
        replay_buffer.reset()

        # end timer
        ct_final = time.time()

        # save episode
        cts[ep] = ct_final - ct_initial

        #print('it: {:3d}, fht: {:2.2f}'.format(it+1, fht))

        # update plots
        if live_plot and (ep + 1) % 1 == 0:
            policy = compute_table_det_policy_1d(env, actor)
            update_det_policy_1d_figure(env, policy, line_actor)

    # add learning results
    data['cts'] = cts

    save_data(data, rel_dir_path)
    return data


def dpg_replay(env, **kwargs):

    # get dir path
    rel_dir_path = get_dpg_dir_path(env, agent='dpg-replay', **kwargs)

    # load results
    load = kwargs['load']
    if load:
        return load_data(rel_dir_path)

    # set seed
    seed = kwargs['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initialize models and optimizer

        # update actor parameters
        #update_actor_critic_naive(actor, actor_optimizer, critic, batch['states'])

        # end timer
        #ct_final = time.time()

        # save episode
        cts[ep] = ct_final - ct_initial

        #print('it: {:3d}, fht: {:2.2f}'.format(it+1, fht))

        # update plots
        if live_plot and (it + 1) % 1 == 0:
            policy = compute_table_det_policy_1d(env, actor)
            update_det_policy_1d_figure(env, policy, line_actor)

    # add learning results
    data['cts'] = cts

    save_data(data, rel_dir_path)
    return data


def dpg_batch(env, **kwargs):

    # get dir path
    rel_dir_path = get_dpg_dir_path(env, agent='dpg-batch', **kwargs,)

    # load results
    load = kwargs['load']
    if load:
        return load_data(rel_dir_path)

    # set seed
    seed = kwargs['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


    # initialize models and optimizer
    actor, actor_optimizer, critic = initialize_actor_and_critic(env, **kwargs)

    # initialize replay buffer
    batch_size = kwargs['batch_size']
    replay_buffer = ReplayBuffer(
        size=int(1e5), state_dim=env.state_space_dim, action_dim=env.action_space_dim,
    )

    # save algorithm relevant parameters
    data = {
        key: value for key, value in kwargs.items() if key in [
            'gamma',
            'd_hidden_layer',
            'n_layers',
            'n_steps_episode_lim',
            'batch_size',
            'lr_actor',
            'n_iterations',
            'seed',
    ]}
    data['actor'] = actor
    data['rel_dir_path'] = rel_dir_path
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-it{}'.format(0))

    # ct per iterations
    n_iterations = kwargs['n_iterations']
    cts = np.empty(n_iterations, dtype=np.float32)
    cts.fill(np.nan)

    # test policy
    test = kwargs['test']
    policy_opt = kwargs['policy_opt']
    if test:
        test_batch_size = kwargs['test_batch_size']
        test_freq_iterations = kwargs['test_freq_iterations']
        test_policy = TestPolicy(test_batch_size, test_freq_iterations, freq_type='it')
        test_policy.test_actor_model(env, actor, policy_opt)

    # get initial state
    state_init = env.state_init.copy()

    # initialize figures if plot:
    live_plot = kwargs['live_plot']
    if live_plot:
        policy = compute_table_det_policy_1d(env, actor)
        line_actor = initialize_det_policy_1d_figure(env, policy, policy_opt)

    for it in range(n_iterations):

        # start timer
        ct_initial = time.time()

        # sample trajectories
        sample_trajectories_buffer_vectorized(env, actor, replay_buffer,
                                              batch_size, kwargs['n_steps_episode_lim'])

        # sample minibatch of transition uniformlly from the replay buffer
        batch = replay_buffer.sample_batch(batch_size)

        # update actor parameters
        update_parameters(actor, actor_optimizer, critic, batch['states'])

        # reset replay buffer
        replay_buffer.reset()

        # end timer
        ct_final = time.time()

        # save episode
        cts[ep] = ct_final - ct_initial

        #print('it: {:3d}, fht: {:2.2f}'.format(it+1, fht))

        # update plots
        if live_plot and (ep + 1) % 1 == 0:
            policy = compute_table_det_policy_1d(env, actor)
            update_det_policy_1d_figure(env, policy, line_actor)

    # add learning results
    data['cts'] = cts

    save_data(data, rel_dir_path)
    return data


def dpg_replay(env, **kwargs):

    # get dir path
    rel_dir_path = get_dpg_dir_path(env, agent='dpg-replay', **kwargs)

    # load results
    load = kwargs['load']
    if load:
        return load_data(rel_dir_path)

    # set seed
    seed = kwargs['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initialize models and optimizer
    actor, actor_optimizer, critic = initialize_actor_and_critic(env, **kwargs)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(
        size=int(1e2), state_dim=env.state_space_dim, action_dim=env.action_space_dim,
    )

    # save algorithm relevant parameters
    data = {
        key: value for key, value in kwargs.items() if key in [
            'gamma',
            'd_hidden_layer',
            'n_layers',
            'n_steps_episode_lim',
            'batch_size',
            'lr_actor',
            'n_episodes',
            'seed',
    ]}
    data['actor'] = actor
    data['rel_dir_path'] = rel_dir_path
    save_data(data, rel_dir_path)

    # save models initial parameters
    save_model(actor, rel_dir_path, 'actor_n-it{}'.format(0))

    # ct per iterations
    n_episodes = kwargs['n_episodes']
    cts = np.empty(n_episodes, dtype=np.float32)
    cts.fill(np.nan)

    # test policy
    test = kwargs['test']
    policy_opt = kwargs['policy_opt']
    if test:
        test_batch_size = kwargs['test_batch_size']
        test_freq_episodes = kwargs['test_freq_episodes']
        test_policy = TestPolicy(test_batch_size, test_freq_episodes, freq_type='ep')
        test_policy.test_actor_model(env, actor, policy_opt)

    # get initial state
    state_init = env.state_init.copy()

    # initialize figures if plot:
    live_plot = kwargs['live_plot']
    if live_plot:
        policy = compute_table_det_policy_1d(env, actor)
        line_actor = initialize_det_policy_1d_figure(env, policy, policy_opt)

    # sample trajectories
    for ep in range(n_episodes):

        # start timer
        ct_initial = time.time()

        # initialization
        state = env.reset()

        # reset trajectory return
        ep_return = 0

        # terminal state flag
        done = False

        # sample trajectory
        for k in np.arange(kwargs['n_steps_episode_lim']):

            # interrupt if we are in a terminal state
            if done:
                break

            # sample action
            action = actor.forward(torch.FloatTensor(state)).detach().numpy()

            # env step
            next_state, r, done, _ = env.step(state, action)

            # store tuple
            #replay_buffer.store(state, action, r, next_state, done)

            # sample minibatch of transition uniformlly from the replay buffer
            #batch = replay_buffer.sample_batch(kwargs['batch_size'])

            # update actor parameters
            #update_parameters(actor, actor_optimizer, critic, batch)
            update_parameters(actor, actor_optimizer, critic, next_state)

            # update state
            state = next_state

        # end timer
        ct_final = time.time()

        # save statistics
        cts[ep] = ct_final - ct_initial

        # test actor model
        if test and (ep + 1) % test_freq_episodes == 0:
            test_policy.test_actor_model(env, actor, policy_opt)

        # update plots
        if live_plot and (ep + 1) % 1 == 0:
            policy = compute_table_det_policy_1d(env, actor)
            update_det_policy_1d_figure(env, policy, line_actor)

    # add learning results
    data['cts'] = cts

    # add test results
    if test:
        data = data | test_policy.get_dict()

    save_data(data, rel_dir_path)
    return data

def load_backup_models(data, ep=0):
    actor = data['actor']
    critic = data['critic']
    rel_dir_path = data['rel_dir_path']
    try:
        load_model(actor, rel_dir_path, file_name='actor_n-ep{}'.format(ep))
        load_model(critic, rel_dir_path, file_name='critic_n-ep{}'.format(ep))
    except FileNotFoundError as e:
        print('The episode {:d} has no backup '.format(ep))

