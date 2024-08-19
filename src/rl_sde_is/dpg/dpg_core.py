from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect

from rl_sde_is.dpg.replay_memories import ReplayMemory
from rl_sde_is.dpg.dpg_utils import DeterministicPolicy, QValueFunction
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.approximate_methods import *
from rl_sde_is.utils.plots import *
from rl_sde_is.utils.path import *
#from rl_sde_is.dpg_utils import DeterministicPolicy, ValueFunction, AValueFunction

def initialize_actor_and_critic(env, n_layers, d_hidden_layer, lr, value_function_opt, policy_opt):

    # dimensions of the state and action space
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]

    # initialize actor representations
    actor = DeterministicPolicy(
        state_dim=env.d,
        action_dim=env.d,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh(),
    )

    # initialize critic representations
    critic_q = QValueFunction(
        state_dim=env.d,
        action_dim=env.d,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh()
    )
    critic_q = train_critic_from_dp(env, critic_q, value_function_opt, policy_opt, load=True)
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
    #actor_optimizer = optim.SGD(actor.parameters(), lr=lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)

    return actor, actor_optimizer, critic_q

def update_parameters_actor(actor, actor_optimizer, critic, states, fht_estimate=1.0):

    # unpack tuples in batch
    states = torch.tensor(states)

    # get batch size
    batch_size = states.shape[0]

    # actor loss
    actor_loss = - critic.forward(states, actor.forward(states)).mean()

    # reset and compute actor gradients
    actor_optimizer.zero_grad()
    actor_loss.backward()

    # scale learning rate
    actor_optimizer.param_groups[0]['lr'] *= fht_estimate

    #update parameters
    actor_optimizer.step()

    # re-scale learning rate back
    actor_optimizer.param_groups[0]['lr'] /= fht_estimate

    return actor_loss

def update_parameters_actor_critic_naive(actor, actor_optimizer, critic, critic_optimizer, experience):

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

    # q value for the given pairs of states and actions
    q_vals = critic.forward(states, actions)

    with torch.no_grad():

        # next actions following the actor
        next_actions = actor.forward(next_states)

        # q value for the next-state-next-action pair
        q_vals_next = critic.forward(next_states, next_actions)

        d = torch.where(done, 1., 0.)
        targets = rewards + (1. - d) * q_vals_next

    # critic loss
    critic_loss = (q_vals - targets).pow(2).mean()

    # reset and compute critic gradients and update parameters
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # freeze q-networks to save computational effort 
    for param in critic.parameters():
        param.requires_grad = False

    # 2) actor loss
    actor_loss = - fht_estimate * critic.forward(states, actor.forward(states)).mean()

    # reset actor and compute actor gradients and update parameters
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # unfreeze q-network to save computational effort 
    for param in critic.parameters():
        param.requires_grad = True

def update_parameters_actor_critic_polyak(env, actor, actor_target, actor_optimizer, critic,
                                          critic_target, critic_optimizer, batch, gamma, polyak):

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
        next_actions = actor_target.forward(next_states)
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


def dpg_actor_critic_naive(env, **kwargs):

    # get dir path
    dir_path = get_dpg_dir_path(env, agent='dpg-naive', **kwargs)

    # load results
    if kwargs['load']:
        return load_data(dir_path)

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
    data['dir_path'] = dir_path
    save_data(data, dir_path)

    # save models initial parameters
    save_model(actor, dir_path, 'actor_n-ep{}'.format(0))
    save_model(critic, dir_path, 'critic_n-ep{}'.format(0))

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
            save_model(actor, dir_path, 'actor_n-ep{}'.format(ep + 1))
            save_model(critic, dir_path, 'critic_n-ep{}'.format(ep + 1))

        # update plots
        if live_plot and (ep + 1) % 1 == 0:
            q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
            v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
            update_actor_critic_1d_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                           a_table, policy_actor, policy_critic, lines)


    # add learning results
    data['cts'] = cts

    save_data(data, dir_path)
    return data

def dpg_optimal(env, expectation_type, gamma, n_layers, d_hidden_layer, estimate_mfht,
                batch_size, mini_batch_size, lr, n_grad_iterations, seed=None, n_steps_lim=1e6,
                backup_freq=None, value_function_opt=None, policy_opt=None, live_plot_freq=False, load=False):

    # get dir path
    dir_path = get_dpg_optimal_dir_path(
        env,
        agent='dpg-optimal',
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
        estimate_mfht=estimate_mfht,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        lr=lr,
        n_grad_iterations=n_grad_iterations,
        seed=seed,
    )

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = RecordEpisodeStatisticsVect(env, batch_size)#, track_l2_error)

    # initialize models and optimizer
    actor, actor_optimizer, critic = initialize_actor_and_critic(
        env, n_layers, d_hidden_layer, lr, value_function_opt, policy_opt,
    )

    # freeze q-networks to save computational effort 
    for param in critic.parameters():
        param.requires_grad = False

    # initialize replay buffer
    memory = ReplayMemory(size=int(1e6), state_dim=env.d, action_dim=env.d)

    # save algorithm relevant parameters
    data = {
        'gamma': gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'batch_size': batch_size,
        'lr': lr,
        'n_grad_iterations': n_grad_iterations,
        'seed': seed,
        'actor': actor,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # save models initial parameters
    save_model(actor, dir_path, 'actor_n-it{}'.format(0))

    # create object to store the is statistics of the learning
    is_stats = ISStatistics(
        eval_freq=1,
        eval_batch_size=batch_size,
        n_grad_iterations=n_grad_iterations,
        track_loss=True,
        track_ct=True,
    )
    keys_chosen = [
        'max_lengths', 'mean_fhts', 'var_fhts',
        'mean_returns', 'var_returns',
        'mean_I_us', 'var_I_us', 're_I_us',
        'losses',
        'cts',
    ]

    # ct per iterations
    cts = np.full(n_grad_iterations, np.nan)

    # initialize figures if plot:
    if live_plot_freq:
        states = torch.FloatTensor(env.state_space_h)
        policy = compute_det_policy_actions(env, actor, states)
        line_actor = initialize_det_policy_1d_figure(env, policy, policy_opt)

    for it in range(n_grad_iterations):

        # start timer
        ct_initial = time.time()

        # sample trajectories
        sample_trajectories_buffer_vect(env, actor, memory, batch_size, n_steps_lim)

        # sample minibatch of transition uniformlly from the replay buffer
        batch = memory.sample_batch(mini_batch_size)#batch_size)
        #batch = replay_buffer.sample_batch(replay_buffer.size)
        fht_estimate = memory.estimate_episode_length() if estimate_mfht else 1.

        # update actor parameters
        loss = update_parameters_actor(actor, actor_optimizer, critic,
                                       batch['states'], fht_estimate)

        # reset replay buffer
        memory.reset()

        # save stats
        ct = time.time() - ct_initial

        # save and log epoch 
        env.statistics_to_numpy()
        is_stats.save_epoch(it, env, loss=loss.detach().numpy(), loss_var=np.nan, ct=ct)
        is_stats.log_epoch(it)

        #print('it: {:3d}, fht: {:2.2f}'.format(it+1, fht))

        # update plots
        if live_plot_freq and (it + 1) % live_plot_freq == 0:
            states = torch.FloatTensor(env.state_space_h)
            policy = compute_det_policy_actions(env, actor, states)
            update_det_policy_1d_figure(env, policy, line_actor)

    stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
    data = data | stats_dict
    save_data(data, dir_path)
    return data

def dpg_actor_critic_batch(env, gamma=1., n_layers=3, d_hidden_layer=32, batch_size=1000, lr_actor=1e-3, lr_critic=1e-3,
                           n_steps_lim=1000, n_grad_iterations=1000, seed=None, backup_freq=None, estimate_mfht=False,
                           value_function_opt=None, policy_opt=None, live_plot_freq=False, load=False, polyak=0.995):

    # get dir path
    dir_path = get_dpg_dir_path(
        env,
        agent='dpg-actor-critic-batch',
        gamma=gamma,
        n_layers=n_layers,
        d_hidden_layer=d_hidden_layer,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        n_grad_iterations=n_grad_iterations,
        seed=seed,
    )

    # load results
    if load:
        return load_data(dir_path)

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initialize models and optimizers
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    actor = DeterministicPolicy(state_dim=env.d, action_dim=env.d,
                                hidden_sizes=hidden_sizes, activation=nn.Tanh())
    actor_target = deepcopy(actor)

    # initialize critic representations
    critic = QValueFunction(state_dim=env.d, action_dim=env.d,
                            hidden_sizes=hidden_sizes, activation=nn.Tanh())
    critic_target = deepcopy(critic)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(size=int(1e6), state_dim=env.d, action_dim=env.d)

    # save algorithm relevant parameters
    data = {
        'gamma': gamma,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'batch_size': batch_size,
        'lr_actor': lr_actor,
        'lr_critic': lr_critic,
        'n_grad_iterations': n_grad_iterations,
        'seed': seed,
        'actor': actor,
        'critic': critic,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # save models initial parameters
    save_model(actor, dir_path, 'actor_n-it{}'.format(0))
    save_model(critic, dir_path, 'critic_n-it{}'.format(0))

    # ct per iterations
    cts = np.full(n_grad_iterations, np.nan)

    # initialize figures if plot:
    if live_plot_freq:
        q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
        v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
        lines = initialize_actor_critic_1d_figures(
            env, q_table, v_table_actor_critic, v_table_critic, a_table,
            policy_actor, policy_critic, value_function_opt, policy_opt)

    for it in range(n_grad_iterations):

        # start timer
        ct_initial = time.time()

        # sample trajectories
        sample_trajectories_buffer_vect(env, actor, replay_buffer, batch_size, n_steps_lim)

        for _ in range(1):
        #for _ in range(100):

            # sample minibatch of transition uniformlly from the replay buffer
            #batch = replay_buffer.sample_batch(batch_size)
            batch = replay_buffer.sample_batch(replay_buffer.size)

            # update actor and critic parameters
            #update_actor_critic_simple(actor, actor_optimizer, critic, critic_optimizer,
            #                           batch)#, fht_estimate)

            update_parameters_actor_critic_polyak(env, actor, actor_target, actor_optimizer, critic,
                                          critic_target, critic_optimizer, batch, gamma, polyak)

        # reset replay buffer
        replay_buffer.reset()

        # save stats
        cts[it] = time.time() - ct_initial

        print('it: {:3d}'.format(it+1))

        # update plots
        if live_plot_freq and (it + 1) % live_plot_freq == 0:
            q_table, v_table_critic, a_table, policy_critic = compute_tables_critic_1d(env, critic)
            v_table_actor_critic, policy_actor = compute_tables_actor_critic_1d(env, actor, critic)
            update_actor_critic_1d_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                           a_table, policy_actor, policy_critic, lines)


    save_data(data, dir_path)
    return data


def dpg_replay(env, **kwargs):

    # get dir path
    dir_path = get_dpg_dir_path(env, agent='dpg-replay', **kwargs)

    # load results
    load = kwargs['load']
    if load:
        return load_data(dir_path)

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

    save_data(data, dir_path)
    return data





def dpg_replay(env, **kwargs):

    # get dir path
    dir_path = get_dpg_dir_path(env, agent='dpg-replay', **kwargs)

    # load results
    load = kwargs['load']
    if load:
        return load_data(dir_path)

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
    data['dir_path'] = dir_path
    save_data(data, dir_path)

    # save models initial parameters
    save_model(actor, dir_path, 'actor_n-it{}'.format(0))

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
        states = torch.FloatTensor(env.state_space_h)
        policy = compute_det_policy_actions(env, model, states)
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
            states = torch.FloatTensor(env.state_space_h)
            policy = compute_det_policy_actions(env, model, states)
            update_det_policy_1d_figure(env, policy, line_actor)

    # add learning results
    data['cts'] = cts

    # add test results
    if test:
        data = data | test_policy.get_dict()

    save_data(data, dir_path)
    return data

def load_backup_models(data, ep=0):
    actor = data['actor']
    critic = data['critic']
    dir_path = data['dir_path']
    try:
        load_model(actor, dir_path, file_name='actor_n-ep{}'.format(ep))
        load_model(critic, dir_path, file_name='critic_n-ep{}'.format(ep))
    except FileNotFoundError as e:
        print('The episode {:d} has no backup '.format(ep))

