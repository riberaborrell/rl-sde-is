import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import FeedForwardNN
from rl_sde_is.plots import *
from rl_sde_is.replay_buffers import ContinuousReplayBuffer as ReplayBuffer
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

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
    q_vals = critic.forward(torch.hstack((states, actions))).squeeze()

    with torch.no_grad():

        # q value for the corresponding next pair of states and actions (using target networks)
        next_actions = actor_target.forward(next_states).detach()
        q_vals_next = critic_target.forward(torch.hstack((next_states, next_actions))).squeeze()

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
    inputs = torch.hstack((states, actor.forward(states)))
    actor_loss = - critic.forward(inputs).mean()

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


def ddpg(env, gamma=0.99, hidden_size=32, n_layers=3,
         n_episodes=100, n_steps_episode_lim=1000,
         start_steps=0, update_after=1000, update_freq=100, test_freq_episodes=100, backup_freq_episodes=None,
         replay_size=10000, batch_size=512, lr_actor=1e-4, lr_critic=1e-4, test_batch_size=100,
         rho=0.95, seed=None,
         value_function_hjb=None, control_hjb=None, load=False, plot=False):

    # get dir path
    rel_dir_path = get_ddpg_dir_path(
        env,
        agent='ddpg-episodic',
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
    d_in = d_state_space
    d_out = d_action_space
    actor_hidden_sizes = [hidden_size for i in range(n_layers -1)]
    actor = FeedForwardNN(d_in, actor_hidden_sizes, d_out)
    actor_target = FeedForwardNN(d_in, actor_hidden_sizes, d_out)

    # initialize critic representations
    d_in = d_state_space + d_action_space
    d_out = 1
    critic_hidden_sizes = [hidden_size for i in range(n_layers -1)]
    critic = FeedForwardNN(d_in, critic_hidden_sizes, d_out)
    critic_target = FeedForwardNN(d_in, critic_hidden_sizes, d_out)

    # set same parameters actor
    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)

    # set same parameters critic
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)

    # set optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=d_state_space, action_dim=d_action_space,
                                 size=replay_size)

    # initialize figures
    if plot:
        q_table, v_table_critic, a_table, policy_critic = compute_tables_continuous_actions(env, critic)
        v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)
        lines = initialize_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                                a_table, policy_actor, policy_critic,
                                                value_function_hjb, control_hjb)

    # save initial parameters
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
            action = actor.forward(torch.FloatTensor(state)).detach().numpy()

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
                    = test_policy_vectorized(env, actor, batch_size=100, control_hjb=control_hjb)
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

        # save actor and critic models
        if backup_freq_episodes is not None and (ep + 1) % backup_freq_episodes == 0:
            save_model(actor, rel_dir_path, 'actor_n-epi{}'.format(ep + 1))
            save_model(critic, rel_dir_path, 'critic_n-epi{}'.format(ep + 1))

        # update plots
        if plot and (ep + 1) % 1 == 0:
            q_table, v_table_critic, a_table, policy_critic = compute_tables_continuous_actions(env, critic)
            v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)
            update_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                    a_table, policy_actor, policy_critic, lines)

    data = {
        'gamma' : gamma,
        'batch_size' : batch_size,
        'lr_actor' : lr_actor,
        'lr_critic' : lr_critic,
        'n_episodes': n_episodes,
        'seed': seed,
        'replay_size': replay_size,
        'update_after': update_after,
        'returns': returns,
        'time_steps': time_steps,
        'test_batch_size' : test_batch_size,
        'test_mean_returns': test_mean_returns,
        'test_var_returns': test_var_returns,
        'test_mean_lengths': test_mean_lengths,
        'test_u_l2_errors': test_u_l2_errors,
        'actor': actor,
        'critic': critic,
        'rel_dir_path': rel_dir_path,
    }
    save_data(data, rel_dir_path)
    return data

def load_backup_models(actor, critic, rel_dir_path, ep=0):
    load_model(actor, rel_dir_path, file_name='actor_n-epi{}'.format(ep))
    load_model(critic, rel_dir_path, file_name='critic_n-epi{}'.format(ep))


def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize state and action space (plot purposes only)
    env.discretize_state_space(h_state=0.01)
    env.discretize_action_space(h_action=0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run ddpg 
    data = ddpg(
        env=env,
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        replay_size=10000,
        n_steps_episode_lim=10000,
        test_freq_episodes=100,
        backup_freq_episodes=args.backup_freq_episodes,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )

    # plots
    if not args.plot:
        return

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

    # get models
    actor = data['actor']
    critic = data['critic']

    # get backup models
    load_backup_models(actor, critic, data['rel_dir_path'], ep=0)

    # compute tables following q-value model
    q_table, v_table_critic, a_table, policy_critic = compute_tables_continuous_actions(env, critic)

    # compute value function and actions following the policy model
    v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)

    plot_q_value_function(env, q_table)
    plot_value_function_actor_critic(env, v_table_actor_critic, v_table_critic, sol_hjb.value_function)
    plot_advantage_function(env, a_table)
    plot_det_policy_actor_critic(env, policy_actor, policy_critic, sol_hjb.u_opt)


if __name__ == '__main__':
    main()
