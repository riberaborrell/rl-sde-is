import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from approximate_methods import *
from base_parser import get_base_parser
from environments import DoubleWellStoppingTime1D
from models import FeedForwardNN
from plots import *
from replay_buffers import ContinuousReplayBuffer as ReplayBuffer
from utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def get_v_value(critic, state):
    actions = torch.arange(-5, 5+0.01, 0.01).unsqueeze(dim=1)
    states = torch.ones_like(actions) * torch.FloatTensor(state)
    inputs = torch.hstack((states, actions))
    with torch.no_grad():
        q_values = critic.forward(inputs).numpy()
    return np.max(q_values)

def update_parameters(actor, actor_target, actor_optimizer, critic, critic_target,
                      critic_optimizer, rho, batch, gamma):

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
        for target_param, param in zip(actor_target.parameters(), actor.parameters()):
            target_param.data.copy_(target_param.data * rho + param.data * (1. - rho))

        for target_param, param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(target_param.data * rho + param.data * (1. - rho))

    return actor_loss.detach().item(), critic_loss.detach().item()


def ddpg(env, gamma=1., hidden_size=32, n_layers=3, lr_actor=1e-3, lr_critic=1e-3,
         n_episodes=100, n_avg_episodes=10, n_steps_lim=1000,
         replay_size=10000, replay_min_size=1000, batch_size=100, target_update_freq=100,
         rho=0.995, value_function_hjb=None, control_hjb=None, load=False):

    # get dir path
    dir_path = get_ddpg_dir_path(
        env,
        agent='ddpg',
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        n_episodes=n_episodes,
        batch_size=batch_size,
    )

    # load results
    if load:
        data = load_data(dir_path)
        return data

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

    # define list to store results
    returns = np.empty(n_episodes)
    avg_returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    avg_time_steps = np.empty(n_episodes)

    # get initial state
    state_init = env.state_init.copy()

    q_table, v_table_critic, a_table, policy_critic = compute_tables_continuous_actions(env, critic)
    tuples = initialize_q_learning_figures(env, q_table, v_table_critic, a_table, policy_critic,
                                           value_function_hjb, control_hjb)

    # sample trajectories
    for ep in range(n_episodes):

        # initialization
        state = env.reset()

        # reset trajectory return
        ep_return = 0

        # terminal state flag
        complete = False

        # sample trajectory
        for k in np.arange(n_steps_lim):

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
            if replay_buffer.size > replay_min_size and k % target_update_freq:

                #if replay_buffer.size == replay_min_size:
                #    print('Replay buffer is ready to sample!')

                # sample minibatch of transition uniformlly from the replay buffer
                batch = replay_buffer.sample_batch(batch_size)

                # update actor and critic parameters
                actor_loss, critic_loss = update_parameters(actor, actor_target, actor_optimizer,
                                  critic, critic_target, critic_optimizer, rho, batch, gamma)

            # save action and reward
            ep_return += (gamma**k) * r
            #print(ep_return)

            # update state
            state = next_state

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            idx_last_episodes = slice(0, ep + 1)
        else:
            idx_last_episodes = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        returns[ep] = ep_return
        avg_returns[ep] = np.mean(returns[idx_last_episodes])
        time_steps[ep] = k
        avg_time_steps[ep] = np.mean(time_steps[idx_last_episodes])

        # logs
        v_value = get_v_value(critic, state_init)
        if ep % n_avg_episodes == 0:
            msg = 'ep: {:3d}, V(s_init): {:.3f}, run avg return {:2.2f}, ' \
                  'run avg time steps: {:2.2f}'.format(
                    ep,
                    v_value,
                    avg_returns[ep],
                    avg_time_steps[ep],
                )
            print(msg)

            # update plots
            q_table, v_table_critic, a_table, policy_critic \
                    = compute_tables_continuous_actions(env, critic)
            update_q_learning_figures(env, q_table, v_table_critic, a_table, policy_critic, tuples)

    data = {
        'n_episodes': n_episodes,
        'returns': returns,
        'avg_returns': avg_returns,
        'time_steps': time_steps,
        'avg_time_steps': avg_time_steps,
        'actor': actor,
        'critic': critic,
    }
    save_data(dir_path, data)
    return data

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
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        n_steps_lim=args.n_steps_lim,
        target_update_freq=args.target_update_freq,
        batch_size=args.batch_size,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
    )
    returns = data['returns']
    avg_returns = data['avg_returns']
    time_steps = data['time_steps']
    avg_time_steps = data['avg_time_steps']
    actor = data['actor']
    critic = data['critic']

    # compute tables following q-value model
    q_table, v_table_critic, a_table, policy_critic = compute_tables_continuous_actions(env, critic)

    # compute value function and actions following the policy model
    v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)

    plot_returns_episodes(returns, avg_returns)
    plot_time_steps_episodes(time_steps, avg_time_steps)
    plot_q_value_function(env, q_table)
    plot_value_function_actor_critic(env, v_table_actor_critic, v_table_critic, sol_hjb.value_function)
    plot_advantage_function(env, a_table)
    plot_det_policy_actor_critic(env, policy_actor, policy_critic, sol_hjb.u_opt)

if __name__ == '__main__':
    main()
