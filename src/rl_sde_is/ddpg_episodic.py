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

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

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
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

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
                      critic_optimizer, batch, gamma, rho):

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


def ddpg(env, gamma=0.99, d_hidden_layer=256, n_layers=3,
         n_episodes=100, n_steps_episode_lim=1000,
         start_steps=0, update_after=5000, test_freq_episodes=100, backup_freq_episodes=None,
         replay_size=50000, batch_size=512, lr_actor=1e-4, lr_critic=1e-4, test_batch_size=1000,
         rho=0.95, seed=None,
         value_function_hjb=None, control_hjb=None, load=False, plot=False):

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
    pre_train_actor(env, actor)

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
    test_u_l2_errors = np.empty((0), dtype=np.float32)

    # get initial state
    state_init = env.state_init.copy()

    # sample trajectories
    for ep in range(n_episodes):

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

            # get action following the actor
            action = get_action(env, actor, state)

            # env step
            next_state, r, complete = env.step(state, action)
            #print(k, state, action, r, next_state)

            # store tuple
            replay_buffer.store(state, action, r, next_state, complete)

            # if buffer is full enough
            if replay_buffer.size > update_after:

                # sample minibatch of transition uniformlly from the replay buffer
                batch = replay_buffer.sample_batch(batch_size)

                # update actor and critic parameters
                actor_loss, critic_loss = update_parameters(
                    actor, actor_target, actor_optimizer,
                    critic, critic_target, critic_optimizer,
                    batch, gamma, rho,
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
            test_mean_ret, test_var_ret, test_mean_len, test_u_l2_error \
                    = test_policy_vectorized(env, actor, batch_size=test_batch_size,
                                             control_hjb=control_hjb)
            test_mean_returns = np.append(test_mean_returns, test_mean_ret)
            test_var_returns = np.append(test_var_returns, test_var_ret)
            test_mean_lengths = np.append(test_mean_lengths, test_mean_len)
            test_u_l2_errors = np.append(test_u_l2_errors, test_u_l2_error)

            msg = 'ep: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
                  'test mean time steps: {:2.2f}, test u l2 error: {:.2e}'.format(
                ep + 1,
                test_mean_ret,
                test_var_ret,
                test_mean_len,
                test_u_l2_error,
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
            data['test_u_l2_errors'] = test_u_l2_errors
            save_data(data, rel_dir_path)

        # update plots
        if plot and (ep + 1) % 1 == 0:
            print(ep)
            q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic)
            v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)
            update_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                    a_table, policy_actor, policy_critic, lines)

    data['returns'] = returns
    data['time_steps'] = time_steps
    data['test_batch_size'] = test_batch_size
    data['test_mean_returns'] = test_mean_returns
    data['test_var_returns'] = test_var_returns
    data['test_mean_lengths'] = test_mean_lengths
    data['test_u_l2_errors'] = test_u_l2_errors
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


def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta)

    # set action space bounds
    env.action_space_low = -5
    env.action_space_high = 5

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize state and action space (plot purposes only)
    env.discretize_state_space(h_state=0.05)
    env.discretize_action_space(h_action=0.05)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run ddpg 
    data = ddpg(
        env=env,
        gamma=args.gamma,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        replay_size=50000,
        update_after=5000,
        n_steps_episode_lim=1000,
        test_freq_episodes=100,
        test_batch_size=1000,
        backup_freq_episodes=args.backup_freq_episodes,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )

    # plots
    if not args.plot:
        return

    # get models
    actor = data['actor']
    critic = data['critic']

    # get backup models
    load_backup_models(data, ep=args.plot_episode)

    # compute tables following q-value model
    q_table, v_table_critic, a_table, policy_critic = compute_tables_critic(env, critic)

    # compute value function and actions following the policy model
    v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)

    plot_q_value_function(env, q_table)
    plot_value_function_actor_critic(env, v_table_actor_critic, v_table_critic, sol_hjb.value_function)
    plot_advantage_function(env, a_table)
    plot_det_policy_actor_critic(env, policy_actor, policy_critic, sol_hjb.u_opt)

    # plot moving averages for each episode
    returns = data['returns']
    run_mean_returns = compute_running_mean(returns, 10)
    run_var_returns = compute_running_variance(returns, 10)
    time_steps = data['time_steps']
    run_mean_time_steps = compute_running_mean(time_steps, 10)
    plot_run_mean_returns_with_error_episodes(run_mean_returns, run_var_returns)
    plot_time_steps_episodes(time_steps, run_mean_time_steps)

    # plot expected values for each epoch
    test_mean_returns = data['test_mean_returns']
    test_var_returns = data['test_var_returns']
    test_mean_lengths = data['test_mean_lengths']
    test_u_l2_errors = data['test_u_l2_errors']
    plot_expected_returns_with_error_epochs(test_mean_returns, test_var_returns)
    plot_det_policy_l2_error_epochs(test_u_l2_errors)



if __name__ == '__main__':
    main()
