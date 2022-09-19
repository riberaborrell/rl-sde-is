import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import FeedForwardNN
from rl_sde_is.replay_buffers import DiscreteReplayBuffer as ReplayBuffer
from rl_sde_is.approximate_methods import *
from rl_sde_is.tabular_methods import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def pre_train_model(env, optimizer, model):

    # batch size
    batch_size = 10*3

    # number of iterations
    n_iterations = 10**3

    # data points
    states = (env.lb - env.rb) * torch.rand(batch_size, 1) + env.rb

    # targets
    q_values_target = torch.zeros((batch_size, env.n_actions))

    # train
    for i in range(n_iterations):

        # compute q values
        q_values = model.forward(states)

        # compute mse loss
        loss = ((q_values - q_values_target)**2).mean()

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def update_parameters(optimizer, model, target_model, batch, gamma):

    # unpack tuples in batch
    states = torch.tensor(batch['state'])
    next_states = torch.tensor(batch['next_state'])
    idx_acts = torch.tensor(batch['idx_act'])
    rews = torch.tensor(batch['rew'])
    done = torch.tensor(batch['done'])

    # get batch size
    batch_size = states.shape[0]

    # get q values of state-action pairs following the model
    phi = model.forward(states)
    q_vals = phi[torch.arange(batch_size, dtype=torch.int64), idx_acts]

    # get maximum q values of next states following the target model
    target_phi = target_model.forward(next_states)
    q_vals_next = torch.max(target_phi, axis=1)[0]

    # compute target
    d = torch.where(done, 1., 0.)
    targets = rews + gamma * (1. - d) * q_vals_next

    # Bellman error loss 
    loss = ((q_vals - targets)**2).mean()

    # compute gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().item()

def update_parameters_double_dqn(optimizer, model, target_model, batch, gamma):

    # unpack tuples in batch
    states = torch.tensor(batch['state'])
    next_states = torch.tensor(batch['next_state'])
    idx_acts = torch.tensor(batch['idx_act'])
    rews = torch.tensor(batch['rew'])
    done = torch.tensor(batch['done'])

    # get batch size
    batch_size = states.shape[0]

    # get q values of state-action pairs following the model
    phi = model.forward(states)
    q_vals = phi[torch.arange(batch_size, dtype=torch.int64), idx_acts]

    # get next actions corresponding to the maximum q values for the next state following the taget model
    target_phi = target_model.forward(next_states)
    idx_next_actions = torch.argmax(target_phi, axis=1)[0]

    # get q values of next state- next action pairs following the model
    phi_next = model.forward(next_states)
    q_vals_next = phi_next[torch.arange(batch_size, dtype=torch.int64), idx_next_actions]

    # compute target
    d = torch.where(done, 1., 0.)
    targets = rews + gamma * (1. - d) * q_vals_next

    # Bellman error loss 
    loss = ((q_vals - targets)**2).mean()

    # compute gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().item()

def test_q(env, model, eps_final, n_test_eps=10):
    ep_rets, ep_lens = [], []
    for _ in range(n_test_eps):
        state, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        while not(done):
            _, action = get_epsilon_greedy_discrete_action(env, model, state, eps_final)
            state, rew, done = env.step(state, action)
            ep_ret += rew
            ep_len += 1
        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)
    return np.mean(ep_rets), np.mean(ep_lens)


def dqn(env, gamma=1., hidden_size=32, n_layers=3, lr=1e-3,
        n_epochs=100, steps_per_epoch=5000, target_update_freq=100,
        batch_size=32, eps_final=0.05,
        replay_size=25000, steps_before_training=1000, value_function_hjb=None, control_hjb=None,
        load=False):

    # get dir path
    dir_path = get_dqn_dir_path(
        env,
        agent='dqn-replay',
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # load results
    if load:
        data = load_data(dir_path)
        return data

    # total number of time steps
    total_steps = n_epochs * steps_per_epoch + steps_before_training
    finish_decay = int(total_steps * 0.75)

    # initialize qvalue and target qvalue representations
    d_in = env.state_space_dim
    d_out = env.n_actions
    hidden_sizes = [hidden_size for i in range(n_layers -1)]
    model = FeedForwardNN(d_in, hidden_sizes, d_out)
    target_model = FeedForwardNN(d_in, hidden_sizes, d_out)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # set q-value model null in the target set
    pre_train_model(env, optimizer, model)

    # set same parameters
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(param.data)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=env.state_space_dim, size=replay_size)

    # preallocate info
    losses = np.empty(n_epochs)
    losses.fill(np.nan)
    returns = np.empty(n_epochs)
    returns.fill(np.nan)
    test_returns = np.empty(n_epochs)
    test_returns.fill(np.nan)
    time_steps = np.empty(n_epochs)
    time_steps.fill(np.nan)
    test_time_steps = np.empty(n_epochs)
    test_time_steps.fill(np.nan)

    # initialize figures
    images, lines = initialize_figures(env, model, n_epochs, value_function_hjb, control_hjb)

    state, rew, done, epsilon, ep_ret, ep_len = env.reset(), 0, False, 1, 0, 0
    epoch, epoch_losses, epoch_rets, epoch_lens = 0, [], [], []

    for k in range(total_steps):

        # get action
        action_idx, action = get_epsilon_greedy_discrete_action(env, model, state, epsilon)

        # step dynamics forward
        next_state, rew, done = env.step(state, action)

        # store tuple in buffer
        replay_buffer.store(state, action_idx, rew, next_state, done)

        # update state
        state = next_state.copy()

        # update episode return and length
        ep_ret += rew
        ep_len += 1

        if done:
            epoch_rets.append(ep_ret)
            epoch_lens.append(ep_len)
            state, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if k > steps_before_training:
            batch = replay_buffer.sample_batch(batch_size)
            step_loss = update_parameters(optimizer, model, target_model, batch, gamma)
            #step_loss = update_parameters_double_dqn(optimizer, model, target_model, batch, gamma)
            epoch_losses.append(step_loss)

        if k % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

            # update figures
            update_figures(env, model, returns, time_steps, test_returns,
                           test_time_steps, images, lines)


        epsilon = 1 + (eps_final- 1)*min(1, k/finish_decay)

        # at the end of each epoch, evaluate the agent
        if (k - steps_before_training) % steps_per_epoch == 0 and (k - steps_before_training)>0:
            epoch = (k - steps_before_training) // steps_per_epoch
            test_ep_ret, test_ep_len = test_q(env, model, eps_final)

            losses[epoch] = np.mean(epoch_losses)
            returns[epoch] = np.mean(epoch_rets)
            time_steps[epoch] = np.mean(epoch_lens)
            test_returns[epoch] = test_ep_ret
            test_time_steps[epoch] = test_ep_len
            print(('epoch: %d \t loss: %.3f \t train_ret: %.3f' \
                   + '\t train_len: %.3f \t test_ret: %.3f \t test_len: %.3f ' \
                   + '\t epsilon: %.3f')%
                    (epoch, np.mean(epoch_losses), np.mean(epoch_rets),
                     np.mean(epoch_lens), test_ep_ret, test_ep_len, epsilon))

            # reset list for storing epochs info
            epoch_losses, epoch_rets, epoch_lens = [], [], []

    data = {
        'n_epochs': n_epochs,
        'returns': returns,
        'test_returns': test_returns,
        'time_steps': time_steps,
        'test_time_steps': test_time_steps,
        'model': model,
    }
    save_data(dir_path, data)
    return data

def initialize_figures(env, model, n_epochs, value_function_hjb, control_hjb):
    import matplotlib.cm as cm

    # initialize figure with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]
    ax5, ax6 = axes[:, 2]

    ax1.set_title('Q-value function')

    ax2.set_title('Value function')
    ax2.set_xlim(env.state_space_low, env.state_space_high)
    ax2.set_ylim(-3, 1)

    ax3.set_title('Advantage function')

    ax4.set_title('Greedy policy')
    ax4.set_xlim(env.state_space_low, env.state_space_high)
    ax4.set_ylim(env.action_space_low, env.action_space_high)

    ax5.set_title('Return')
    ax5.set_xlim(0, n_epochs)
    ax5.set_ylim(-10, 1)

    ax6.set_title('Time steps')
    ax6.set_xlim(0, n_epochs)
    ax6.set_ylim(0, 1000)

    plt.ion()

    # epochs and nan array
    epochs = np.arange(n_epochs)
    nan_array = np.empty_like(epochs)
    nan_array.fill(np.nan)

    extent = env.state_space_low - env.h_state / 2, env.state_space_high + env.h_state / 2, \
             env.action_space_low - env.h_action / 2, env.action_space_high + env.h_action / 2

    # compute tables
    q_table, v_table, a_table, greedy_actions = compute_tables_discrete_actions(env, model)

    # q-value function
    im1 = ax1.imshow(
        q_table.T,
        origin='lower',
        extent=extent,
        cmap=cm.viridis,
        aspect='auto',
    )

    # value function
    value_function_line = ax2.plot(env.state_space_h, v_table)[0]
    ax2.plot(env.state_space_h, -value_function_hjb)

    # advantage function
    im2 = ax3.imshow(
        a_table.T,
        origin='lower',
        extent=extent,
        cmap=cm.plasma,
        aspect='auto',
    )
    # control
    policy_line = ax4.plot(env.state_space_h, greedy_actions)[0]
    ax4.plot(env.state_space_h, control_hjb)

    # returns
    returns_line = ax5.plot(epochs, nan_array)[0]
    test_returns_line = ax5.plot(epochs, nan_array)[0]

    # time steps
    time_steps_line = ax6.plot(epochs, nan_array)[0]
    test_time_steps_line = ax6.plot(epochs, nan_array)[0]

    plt.show()
    images = (im1, im2)
    lines = (value_function_line, policy_line, returns_line, test_returns_line,
             time_steps_line, test_time_steps_line)
    return images, lines

def update_figures(env, model, returns,
                   time_steps, test_returns, test_time_steps, images, lines):
    # eochs
    epochs = np.arange(returns.shape[0])

    # unpack images and lines
    im1, im2 = images
    value_function_line, policy_line, returns_line, test_returns_line, \
        time_steps_line, test_time_steps_line = lines

    # compute tables
    q_table, v_table, a_table, greedy_actions = compute_tables_discrete_actions(env, model)

    # update plots
    im1.set_data(q_table.T)
    im2.set_data(a_table.T)
    value_function_line.set_data(env.state_space_h, v_table)
    policy_line.set_data(env.state_space_h, greedy_actions)
    returns_line.set_data(epochs, returns)
    test_returns_line.set_data(epochs, test_returns)
    time_steps_line.set_data(epochs, time_steps)
    test_time_steps_line.set_data(epochs, test_time_steps)

    # update figure frequency
    plt.pause(0.01)

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize action space
    env.discretize_action_space(args.h_action)

    # discretize state space (plot purposes only)
    env.discretize_state_space(h_state=0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run dqn with replay buffer
    data = dqn(
        env=env,
        gamma=args.gamma,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.n_steps_per_epoch,
        target_update_freq=args.target_update_freq,
        lr=args.lr,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
    )

    # compute q-value function, value function, advantage function and greedy actions
    q_table, v_table, a_table, greedy_actions = compute_tables_discrete_actions(env, data['model'])

    # plot v function
    x = env.state_space_h.squeeze()
    plt.plot(x, v_table)
    plt.show()

    # plot control
    plt.plot(x, greedy_actions)
    plt.show()

    # plot returns
    plt.figure(figsize=(12, 8))
    plt.plot(data['returns'])
    plt.plot(data['test_returns'])
    plt.ylabel('Total Returns')
    plt.xlabel('Episodes')
    plt.show()

    # plot time steps
    plt.figure(figsize=(12, 8))
    plt.plot(data['time_steps'])
    plt.plot(data['test_time_steps'])
    plt.ylabel('Total Time steps')
    plt.xlabel('Episodes')
    plt.show()

if __name__ == '__main__':
    main()
