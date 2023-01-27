import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

class DeterministicPolicy(nn.Module):
    def __init__(self, d_in, hidden_size, d_out, activation=torch.tanh):
        super(DeterministicPolicy, self).__init__()

        # input, hidden and output dimensions
        self.d_in = d_in
        self.hidden_size = hidden_size
        self.d_out = d_out

        # mean and sigma share the same first layer
        self.linear1 = nn.Linear(d_in, hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_out)

        # activation function
        self.activation = activation

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = torch.tanh(self.linear1(x))
        return self.linear2(x)

def cumsum_list(x):
    x = np.array(x)
    return x[::-1].cumsum()[::-1]

def cumsum_numpy(x):
    return x[::-1].cumsum()[::-1]

def cumsum_torch(x):
    return torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z
def discount_cumsum_torch(x, gamma):
    n = x.shape[0]
    y = gamma**torch.arange(n)
    z = torch.zeros_like(x, dtype=torch.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def normalize_advs_trick(x):
    return (x - np.mean(x))/(np.std(x) + 1e-8)

def reinforce(env, gamma=0.99, lr=0.01, n_episodes=2000,
              batch_size=10):


    # initialize policy
    d_state_space = env.state_space_dim
    d_action_space = env.action_space_dim
    model = DeterministicPolicy(d_in=d_state_space, hidden_size=32, d_out=d_action_space)

    # preallocate lists to hold results
    batch_states = np.empty(0, dtype=np.float32)
    batch_actions = np.empty(0, dtype=np.float32)
    batch_actions_tensor = torch.empty((0, 1), dtype=torch.float32)
    batch_det_int = torch.empty(0, dtype=torch.float32)
    batch_brownian_increments = torch.empty((0, 1), dtype=torch.float32)
    batch_discounted_returns = []
    batch_discounted_returns_tensor = torch.empty(0, dtype=torch.float32)
    batch_counter = 0
    total_returns = []
    total_time_steps = []

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    for ep in np.arange(n_episodes):

        # reset state
        state = env.reset()

        # preallocate rewards for the episode
        ep_rewards = []
        ep_rewards_tensor = torch.empty(0, dtype=torch.float32)

        # time step
        k = 0

        complete = False
        while complete == False:

            # save state
            batch_states = np.concatenate((batch_states, state.copy()))

            # get action following policy
            action_tensor = model.forward(state)
            action = action_tensor.detach().numpy()

            # next step
            new_state, r, complete = env.step(state, action)
            k += 1

            # re compute reward as tensor
            r_tensor = torch.where(
                torch.tensor(complete),
                torch.zeros(1),
                - 0.5 * torch.pow(action_tensor, 2) * env.dt - env.dt
            )

            # save action and brownian increments in batch
            batch_actions = np.concatenate((batch_actions, action))
            batch_actions_tensor = torch.vstack((batch_actions_tensor, action_tensor))
            #batch_det_int = torch.cat(
            #    (batch_det_int, 0.5 * (torch.linalg.norm(action_tensor) ** 2).reshape(1,))
            #)
            batch_brownian_increments = torch.vstack(
                (batch_brownian_increments, torch.tensor(info['dbt']))
            )

            # save rewards in trajectory
            ep_rewards.append(r)
            ep_rewards_tensor = torch.cat(
                (ep_rewards_tensor, r_tensor)
            )

        # update batch data
        batch_discounted_returns.extend(cumsum_list(ep_rewards))
        batch_discounted_returns_tensor = torch.cat(
            (batch_discounted_returns_tensor, cumsum_torch(ep_rewards_tensor))
        )
        batch_counter += 1
        total_returns.append(sum(ep_rewards))
        total_time_steps.append(k)

        # update network if batch is complete 
        if batch_counter == batch_size:

            # reset gradients ..
            optimizer.zero_grad()

            # tensor states, actions and rewards
            n_steps = batch_states.shape[0]
            batch_states = torch.FloatTensor(batch_states)

            # discounted returns
            #batch_discounted_returns = normalize_advs_trick(batch_discounted_returns)
            batch_discounted_returns = torch.FloatTensor(np.array(batch_discounted_returns))

            # compute grad log probability transition function
            grad_log_probs = torch.matmul(
                torch.unsqueeze(batch_actions_tensor, 1),
                torch.unsqueeze(batch_brownian_increments, 2),
            ).reshape(n_steps,)

            # calculate loss
            loss = - (batch_discounted_returns_tensor + batch_discounted_returns * grad_log_probs).mean()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # reset batch
            batch_states = np.empty(0, dtype=np.float32)
            batch_actions = np.empty(0, dtype=np.float32)
            batch_actions_tensor = torch.empty((0, 1), dtype=torch.float32)
            batch_det_int = torch.empty(0, dtype=torch.float32)
            batch_brownian_increments = torch.empty((0, 1), dtype=torch.float32)
            batch_discounted_returns = []
            batch_discounted_returns_tensor = torch.empty(0, dtype=torch.float32)
            batch_counter = 0

            # print running average
            run_avg_msg = 'ep: {}, run avg returns: {:.2f}, run avg time steps: {:.2f}'.format(
                ep + 1,
                np.mean(total_returns[-batch_size:]),
                np.mean(total_time_steps[-batch_size:]),
            )
            print(run_avg_msg)

    return total_returns, total_time_steps, model

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # initial state sampled
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # run reinforce
    returns, time_steps, model = reinforce(
        env=env,
        gamma=1.0,
        lr=args.alpha,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
    )

    # plot deterministic policy
    h = 0.01
    observation_space_h = np.arange(-2, 2+h, h, dtype=np.float32)
    Nh = observation_space_h.shape[0]
    policy_h = model.forward(observation_space_h.reshape(Nh, 1)).detach().numpy()[:, 0]
    plt.figure(figsize=(12, 8))
    plt.plot(observation_space_h, policy_h)
    plt.ylabel('Deterministic policy')
    plt.xlabel('State space')
    plt.show()

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
