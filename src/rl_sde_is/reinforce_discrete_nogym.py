import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_sde_is.base_parser import get_base_parser

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

class Policy(nn.Module):
    def __init__(self, d_in, hidden_size, d_out, activation=torch.tanh):
        super(Policy, self).__init__()

        # input, hidden and output dimensions
        self.d_in = d_in
        self.hidden_size = hidden_size
        self.d_out = d_out

        # two linear layers
        self.linear1 = nn.Linear(d_in, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, d_out)

        # activation function
        self.activation = activation

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        action_scores = self.linear3(x)
        return F.softmax(action_scores, dim=-1)

def discount_cumsum_numpy(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def discount_cumsum(x, gamma):
    n = x.shape[0]
    y = gamma**torch.arange(n)
    z = torch.zeros_like(x, dtype=torch.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def normalize_advs_trick(x):
    return (x - np.mean(x))/(np.std(x) + 1e-8)

def env_step(state, action):

    # sde parameters
    beta = torch.tensor(1, dtype=torch.float32)
    sigma = torch.sqrt(2.0 / beta)
    gradient = lambda x: 4 * x * (x**2 - 1)

    # Euler-Marujama discretization
    dt = torch.tensor(0.01, dtype=torch.float32)

    # target set
    lb, rb = 1, 3

    # brownian increment
    dbt = torch.tensor(torch.sqrt(dt) * torch.randn(1), dtype=torch.float32)

    # sde update
    new_state = state + (- gradient(state) + sigma * action) * dt + sigma * dbt

    # done if new state in the target set
    done = bool(new_state > lb and new_state < rb)

    # compute reward
    #reward = np.where(done, 0, - 0.5 * np.power(action, 2)[0] * self.dt - self.dt)
    if not done:
        reward = - 0.5 * torch.pow(action, 2)[0] * dt - dt
    else:
        reward = 0

    # observation and info
    #observation = self._get_obs()
    #info = self._get_info()

    return new_state, reward, done, dbt

def reinforce(gamma=0.99, lr=0.01, n_episodes=2000,
              batch_size=10, seed=1.):

    # state space
    d_state_space = 1

    # action space
    # discretize actions
    h_action = 1.
    action_space_h = torch.arange(-5, 5+h_action, h_action)
    n_actions = action_space_h.shape[0]
    idx_actions = np.arange(n_actions)

    # initialize policy
    model = Policy(d_in=d_state_space, hidden_size=32, d_out=n_actions)

    # preallocate lists to hold results
    batch_states = torch.empty((0, 1), dtype=torch.float32)
    batch_idx_actions = torch.empty(0, dtype=torch.int8)
    batch_discounted_returns = torch.empty(0, dtype=torch.float32)
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
        state = torch.tensor([-1.], dtype=torch.float32)

        # preallocate rewards for the episode
        ep_rewards = torch.empty(0, dtype=torch.float32)

        # time step
        k = 0

        complete = False
        while complete == False:

            # save state
            batch_states = torch.vstack([batch_states, state.clone()])

            # get action following policy
            action_prob_dist = model.forward(state)
            idx_action = torch.multinomial(action_prob_dist, 1)
            action = torch.tensor([action_space_h[idx_action]])

            # next step
            state, r, complete, _ = env_step(state, action)
            k += 1

            # save action and reward
            batch_idx_actions = torch.cat([batch_idx_actions, idx_action])
            ep_rewards = torch.cat([ep_rewards, r])

        # update batch data
        batch_discounted_returns.extend(discount_cumsum(ep_rewards, gamma))
        batch_counter += 1
        total_returns.append(sum(ep_rewards))
        total_time_steps.append(k)

        # update network if batch is complete 
        if batch_counter == batch_size:

            # reset gradients ..
            optimizer.zero_grad()

            # tensor states, actions and rewards
            idx_action_tensor = torch.LongTensor(np.array(batch_idx_actions))
            #batch_discounted_returns = normalize_advs_trick(batch_discounted_returns)
            discounted_returns_tensor = torch.FloatTensor(np.array(batch_discounted_returns))

            # calculate loss
            action_prob_dists = model.forward(state_tensor)
            log_action_prob_dists = torch.log(action_prob_dists)
            log_probs = log_action_prob_dists[np.arange(len(idx_action_tensor)), idx_action_tensor]

            # negative sign
            loss = - (discounted_returns_tensor * log_probs).mean()
            var_loss = (discounted_returns_tensor * log_probs).detach().numpy().var()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # reset batch
            batch_states = []
            batch_idx_actions = []
            batch_discounted_returns = []
            batch_counter = 0

            # print running average
            print('ep: {}, run avg returns: {:.2f}, run avg time steps: {:.2f}, ' \
                  'loss: {:.2f}, var: {:.2f}'.format(
                    ep + 1,
                    np.mean(total_returns[-batch_size:]),
                    np.mean(total_time_steps[-batch_size:]),
                    loss,
                    var_loss,
                    #total_time_steps[ep],
                )
            )

    return total_returns, total_time_steps, model

def main():
    args = get_parser().parse_args()

    # run reinforce
    returns, time_steps, model = reinforce(
        gamma=1.0,
        lr=0.01,
        n_episodes=args.n_episodes_lim,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # plot policy
    h = 0.01
    observation_space_h = np.arange(-2, 2+h, h, dtype=np.float32)
    policy_h = np.empty_like(observation_space_h)
    for idx, state in enumerate(observation_space_h):
        action_prob_dist = model.forward(torch.tensor([state])).detach().numpy()
        idx_action = np.random.choice(env.idx_actions, p=action_prob_dist)
        policy_h[idx] = np.array([env.action_space_h[idx_action]])
    plt.figure(figsize=(12, 8))
    plt.scatter(observation_space_h, policy_h)
    plt.ylabel('Actions sampled following policy')
    plt.xlabel('State space')
    plt.show()

    # plot action probability distributions
    h = 0.1
    observation_space_h = np.arange(-2, 2+h, h, dtype=np.float32)
    n_states_h = observation_space_h.shape[0]
    n_actions = env.n_actions
    action_prob_dists = model.forward(observation_space_h.reshape(n_states_h, 1)).detach().numpy()
    action_prob_dists = action_prob_dists.reshape(n_actions, n_states_h)
    extent = np.min(observation_space_h), np.max(observation_space_h), \
             np.min(env.action_space_h), np.max(env.action_space_h)
    plt.subplots()
    plt.imshow(action_prob_dists, extent=extent)
    plt.xlabel('Discretized state space')
    plt.ylabel('Discretized action space')
    plt.colorbar()
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