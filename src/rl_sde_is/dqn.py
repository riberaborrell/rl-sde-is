import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import FeedForwardNN

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def get_action_idx(env, action):
    return torch.argmin(torch.abs(torch.FloatTensor(env.action_space_h) - action))

def get_epsilon_greedy_action(env, model, epsilon, state):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state)
            breakpoint()
            phi = model.forward(state_tensor)
            breakpoint()
            return np.expand_dims(torch.max(phi).numpy(), axis=0)

    # pick random action (exploration)
    else:
        return np.random.uniform(env.action_space_low, env.action_space_high, (1,))

def update_parameters(optimizer, model, target_model, batch, gamma):

    # unpack tuples in batch
    states, states_next, idx_actions, rewards, done = batch
    states = torch.FloatTensor(states)
    states_next = torch.FloatTensor(states_next)
    idx_actions = torch.LongTensor(idx_actions)
    rewards = torch.FloatTensor(rewards)
    done = torch.BoolTensor(done)

    # get batch size
    batch_size = states.shape[0]

    # q values for the state and action following the model
    phi = model.forward(states)
    q_val = phi[torch.arange(batch_size, dtype=torch.int64), idx_actions]

    # q values for the next state and the best action following the taget model
    target_phi = target_model.forward(states_next)
    q_val_next = torch.max(target_phi, axis=1)[0]

    # Bellman error loss 
    mse_loss = nn.MSELoss()
    loss = mse_loss(q_val, rewards + gamma * q_val_next)

    # compute gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().item()


def dqn(env, hidden_size=32, n_layers=3,
        lr=1e-3, gamma=1., n_episodes=1000,
        batch_size=10, eps_init=1., eps_final=0.05,
        finish_decay=50000):

    # initialize qvalue and target qvalue representations
    d_in = env.state_space_dim
    d_out = env.n_actions
    hidden_sizes = [hidden_size for i in range(n_layers -1)]
    model = FeedForwardNN(d_in, hidden_sizes, d_out)
    target_model = FeedForwardNN(d_in, hidden_sizes, d_out)

    # set same parameters
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(param.data)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # preallocate lists to hold results
    batch_states = np.zeros([batch_size, env.state_space_dim], dtype=np.float32)
    batch_next_states = np.zeros([batch_size, env.state_space_dim], dtype=np.float32)
    batch_actions = np.zeros(batch_size, dtype=np.int64)
    batch_rewards = np.zeros(batch_size, dtype=np.float32)
    batch_done = np.zeros(batch_size, dtype=np.bool)

    batch_discounted_returns = []
    batch_idx = 0

    total_returns = []
    total_time_steps = []

    # set epsilon
    epsilon = eps_init

    for ep in np.arange(n_episodes):

        # reset state
        state = env.reset()

        # preallocate rewards for the episode
        ep_rewards = []

        # time step
        k = 0

        print(k)

        complete = False
        while complete == False:

            # save state
            batch_states[batch_idx] = state.copy()

            # get action following q-values
            action = get_epsilon_greedy_action(env, model, epsilon, state)
            act_idx = get_action_idx(env, action)

            # next step
            new_state, r, complete = env.step(state, action)
            k += 1

            # save action and reward
            batch_actions[batch_idx] = action
            batch_rewards[batch_idx] = r
            batch_done[batch_idx] = complete

            # update states
            state = new_state

            # update epsilon
            #epsilon = 1 + (eps_final - 1)*min(1, t/finish_decay)
            epsilon = 0.

        # update batch data
        batch_discounted_returns.extend(discount_cumsum(ep_rewards, gamma))
        batch_counter += 1
        total_returns.append(sum(ep_rewards))
        total_time_steps.append(k)

        # batch is complete
        if batch_counter == batch_size:

            # update parameters 
            batch = (batch_states, batch_next_states, batch_actions, batch_rewards, batch_done)
            step_loss = update_parameters(optimizer, model, target_model, batch, gamma)

            # update network
            target_model.load_state_dict(model.state_dict())

            # reset batch
            batch_states = []
            batch_actions = []
            batch_discounted_returns = []
            batch_counter = 0

            # print running average
            run_avg_msg = 'ep: {}, run avg returns: {:.2f}, run avg time steps: {:.2f}'.format(
                ep + 1,
                np.mean(total_returns[-batch_size:]),
                np.mean(total_time_steps[-batch_size:]),
            )
            print(run_avg_msg)

    return model

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize action space
    env.discretize_action_space(args.h_action)

    # run dqn 
    #returns, time_steps, model = dqn(
    model = dqn(
        env=env,
        gamma=1.0,
    )

    # compute value function and greedy actions
    obs = torch.arange(-2, 2+0.01, 0.01).unsqueeze(dim=1)
    phi = model.forward(obs).detach()
    value_function = torch.max(phi, axis=1)[0].numpy()
    policy = torch.argmax(phi, axis=1).numpy()
    actions = env.action_space_h[policy]

    # plot v function
    x = obs.squeeze().numpy()
    plt.plot(x, value_function)
    plt.show()

    # plot control
    plt.plot(x, actions)
    plt.show()

    breakpoint()
    window = args.batch_size

    # plot returns
    smoothed_returns = [
        np.mean(returns[i-window:i+1]) if i > window
        else np.mean(returns[:i+1]) for i in range(len(returns))
    ]
    plt.figure(figsize=(12, 8))
    plt.plot(returns)
    plt.plot(smoothed_returns)
    plt.ylabel('Total Returns')
    plt.xlabel('Episodes')
    plt.show()

    # plot time steps
    smoothed_time_steps = [
        np.mean(time_steps[i-window:i+1]) if i > window
        else np.mean(time_steps[:i+1]) for i in range(len(time_steps))
    ]
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps)
    plt.plot(smoothed_time_steps)
    plt.ylabel('Total Time steps')
    plt.xlabel('Episodes')
    plt.show()

if __name__ == '__main__':
    main()
