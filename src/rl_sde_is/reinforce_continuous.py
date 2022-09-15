import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from base_parser import get_base_parser
from environments import DoubleWellStoppingTime1D
from policies import GaussStochPolicy1, GaussStochPolicy2
from plots import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def normalize_advs_trick(x):
    return (x - np.mean(x))/(np.std(x) + 1e-8)

def reinforce(env, gamma=0.99, lr=0.01, n_episodes=2000, n_avg_episodes=10,
              batch_size=10, seed=1.):


    # initialize policy
    #model = Policy(d_in=d_state_space, hidden_size=32, d_out=d_action_space)
    model = GaussStochPolicy1(d_in=env.state_space_dim, hidden_size=32,
                              d_out=env.action_space_dim, seed=seed)
    #s = env.reset()
    #s = torch.tensor(s)
    #print(model.forward(s))

    # preallocate lists to hold results
    batch_states = []
    batch_actions = []
    batch_discounted_returns = []
    batch_counter = 0
    returns = []
    time_steps = []

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    for ep in np.arange(n_episodes):

        # reset state
        state = torch.FloatTensor(env.reset())

        # preallocate rewards for the episode
        ep_rewards = []

        # time step
        k = 0

        complete = False
        while complete == False:

            # save state
            batch_states.append(state.detach())

            # get action following policy
            action = model.sample_action(state)

            # next step
            next_state, r, complete = env.step(state, action)
            k += 1

            # save action and reward
            batch_actions.append(action.detach())
            ep_rewards.append(r)

            # update state
            state = next_state

        # update batch data
        batch_discounted_returns.extend(discount_cumsum(ep_rewards, gamma))
        batch_counter += 1
        returns.append(sum(ep_rewards))
        time_steps.append(k)

        # update network if batch is complete 
        if batch_counter == batch_size:

            # reset gradients ..
            optimizer.zero_grad()

            # tensor states, actions and rewards
            state_tensor = torch.FloatTensor(batch_states).unsqueeze(dim=1)
            action_tensor = torch.FloatTensor(batch_actions).unsqueeze(dim=1)
            #batch_discounted_returns = normalize_advs_trick(batch_discounted_returns)
            discounted_returns_tensor = torch.FloatTensor(batch_discounted_returns).unsqueeze(dim=1)

            # calculate loss
            log_probs = torch.log(model.probability(state_tensor, action_tensor))
            loss = - (discounted_returns_tensor * log_probs[:, 0]).mean()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # reset batch
            batch_states = []
            batch_actions = []
            batch_discounted_returns = []
            batch_counter = 0

            # print running average
            run_avg_msg = 'ep: {}, run avg returns: {:.2f}, run avg time steps: {:.2f}'.format(
                ep + 1,
                np.mean(returns[-batch_size:]),
                np.mean(time_steps[-batch_size:]),
            )
            print(run_avg_msg)

    data = {
        'n_episodes': n_episodes,
        'batch_size': batch_size,
        'returns': returns,
        'time_steps': time_steps,
        #'controls': controls,
        #'losses': losses,
        #'var_losses': var_losses,
        #'avg_time_steps': avg_time_steps,
        #'cts': cts,
        'model': model,
    }
    #save_data(dir_path, data)
    return data

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretized state and action space (for plot purposes only)
    env.discretize_state_space(h_state=0.01)
    env.discretize_action_space(h_action=0.01)

    # run reinforce (stochastic policy gradient)
    data = reinforce(
        env=env,
        gamma=args.gamma,
        lr=args.lr,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    returns = data['returns']
    time_steps = data['time_steps']
    model = data['model']

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # get actions sampled following policy
    states = np.expand_dims(env.state_space_h, axis=1)
    policy = model.sample_action(states)[:, 0]

    # get action probability distributions
    actions = np.expand_dims(env.action_space_h, axis=1)
    action_prob_dists = np.empty((env.n_states, env.n_actions))
    for idx in range(env.n_states):
        action_prob_dists[idx, :] = model.probability(states[idx], actions)[:, 0].detach().numpy()

    breakpoint()

    plot_stoch_policy(env, action_prob_dists, policy, sol_hjb.u_opt)

    # plot mu and sigma
    mu, sigma_sq = model.forward(states)
    mu = mu.detach().numpy()
    sigma_sq = sigma_sq.detach().numpy()

    plt.figure(figsize=(12, 8))
    plt.plot(env.state_space_h, mu[:, 0])
    plt.ylabel('mu')
    plt.xlabel('State space')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(env.state_space_h, sigma_sq[:, 0])
    plt.ylabel('sigma')
    plt.xlabel('State space')
    plt.show()

    window = args.batch_size

    # plot returns
    smoothed_returns = [
        np.mean(returns[i-window:i+1]) if i > window
        else np.mean(returns[:i+1]) for i in range(len(returns))
    ]
    plot_returns_episodes(returns, smoothed_returns)

    # plot time steps
    smoothed_time_steps = [
        np.mean(time_steps[i-window:i+1]) if i > window
        else np.mean(time_steps[:i+1]) for i in range(len(time_steps))
    ]
    plot_time_steps_episodes(time_steps, smoothed_time_steps)


if __name__ == '__main__':
    main()
