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

    # 1) gradient descent step for q-value netz (critic)

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

    # 2) gradient descent step for policy (actor)

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


def ddpg(env, gamma=0.99, hidden_size=32, n_layers=3,
         n_total_steps=100000, n_steps_episode_lim=600,
         start_steps=0, update_after=1000, update_freq=100, policy_freq=60, test_freq=100,
         replay_size=10000, batch_size=512, lr_actor=1e-4, lr_critic=1e-4,
         rho=0.95, seed=None,
         value_function_hjb=None, control_hjb=None, load=False, plot=False):

    # get dir path
    dir_path = get_ddpg_dir_path(
        env,
        agent='ddpg',
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        n_total_steps=n_total_steps,
        seed=seed,
    )

    # load results
    if load:
        data = load_data(dir_path)
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

    # define list to store episodic results
    returns = np.empty((0), dtype=np.float32)
    time_steps = np.empty((0), dtype=np.float32)

    # preallocate lists to store test results
    test_returns = np.empty((0), dtype=np.float32)
    test_time_steps = np.empty((0), dtype=np.float32)

    # reset episode
    state, rew, complete, ep_ret, ep_len, ep = env.reset(), 0, False, 0, 0, 0

    for k in range(n_total_steps):

        # sample action

        # sample action randomly
        if k < start_steps:
            action = np.random.uniform(env.action_space_low, env.action_space_high, (1,))

        # get action following the actor
        else:
            action = actor.forward(torch.FloatTensor(state)).detach().numpy()

        # step dynamics forward
        next_state, rew, complete = env.step(state, action)

        # store tuple
        replay_buffer.store(state, action, rew, next_state, complete)

        # update state
        state = next_state

        # update episode return and length
        ep_ret += rew
        ep_len += 1

        # update step when buffer is full enough
        if k >= update_after:

            # sample minibatch of transition uniformly from the replay buffer
            batch = replay_buffer.sample_batch(batch_size)

            # update actor and critic parameters
            actor_loss, critic_loss = update_parameters(
            #update_parameters(
                actor, actor_target, actor_optimizer,
                critic, critic_target, critic_optimizer,
                batch, gamma, rho,
            )

        # if trajectory is complete
        if complete:
            msg = 'total k: {:3d}, ep: {:3d}, return {:2.2f}, time steps: {:2.2f}'.format(
                    k,
                    ep,
                    ep_ret,
                    ep_len,
                )
            print(msg)
            returns = np.append(returns, ep_ret)
            time_steps = np.append(time_steps, ep_len)

            # reset episode 
            state, rew, complete, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            ep += 1

        # when epoch is finish evaluate the agent
        if (k + 1) % test_freq == 0:

            # test model
            test_ep_ret, test_ep_len = test_policy_vectorized(env, actor, batch_size=10)
            test_returns = np.append(test_returns, test_ep_ret)
            test_time_steps = np.append(test_time_steps, test_ep_len)

            msg = 'total k: {:3d}, test avg return: {:2.2f} \t test avg time steps: {:2.2f} '.format(
                k,
                test_ep_ret,
                test_ep_len,
            )
            print(msg)

            # update plots
            if plot:
                q_table, v_table_critic, a_table, policy_critic = compute_tables_continuous_actions(env, critic)
                v_table_actor_critic, policy_actor = compute_tables_actor_critic(env, actor, critic)
                update_actor_critic_figures(env, q_table, v_table_actor_critic, v_table_critic,
                                        a_table, policy_actor, policy_critic, lines)



    data = {
        'n_total_steps': n_total_steps,
        'n_episodes': ep,
        'returns': returns,
        'time_steps': time_steps,
        'test_time_steps': test_time_steps,
        'test_returns': test_returns,
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
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        n_total_steps=args.n_total_steps,
        seed=args.seed,
        replay_size=10000,
        n_steps_episode_lim=500,
        start_steps=0,
        update_after=1000,
        update_freq=100,
        policy_freq=60,
        test_freq=1000,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )
    returns = data['returns']
    time_steps = data['time_steps']
    avg_returns = compute_smoothed_array(returns, 10)
    avg_time_steps = compute_smoothed_array(time_steps, 10)
    actor = data['actor']
    critic = data['critic']

    # do plots
    if args.plot:

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
