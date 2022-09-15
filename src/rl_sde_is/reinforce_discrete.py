import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environments import DoubleWellStoppingTime1D
from base_parser import get_base_parser
from plots import *
from policies import DiscreteStochPolicy as Policy

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

def reinforce(env, gamma=0.99, lr=0.01, n_episodes=2000,
              batch_size=10, seed=1.):

    # fix seed
    if seed is not None:
        np.random.seed(seed)

    # initialize policy
    model = Policy(d_in=env.state_space_dim, hidden_size=32, d_out=env.n_actions)
    #s = env.reset()
    #s = torch.tensor(s)
    #print(model.forward(s))

    # preallocate lists to hold results
    batch_states = []
    batch_idx_actions = []
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
        state = env.reset()

        # preallocate rewards for the episode
        ep_rewards = []

        # time step
        k = 0

        complete = False
        while complete == False:

            # save state
            batch_states.append(state.copy())

            # get action following policy
            action_prob_dist = model.forward(state).detach().numpy()
            idx_action = np.random.choice(np.arange(env.n_actions), p=action_prob_dist)
            action = np.array([env.action_space_h[idx_action]])

            # next step
            state, r, complete = env.step(state, action)
            k += 1

            # save action and reward
            batch_idx_actions.append(idx_action)
            ep_rewards.append(r)

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
            state_tensor = torch.FloatTensor(np.array(batch_states))
            idx_action_tensor = torch.LongTensor(np.array(batch_idx_actions))
            batch_discounted_returns = normalize_advs_trick(batch_discounted_returns)
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
                    np.mean(returns[-batch_size:]),
                    np.mean(time_steps[-batch_size:]),
                    loss,
                    var_loss,
                    #total_time_steps[ep],
                )
            )

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

    # discretize action space
    env.discretize_action_space(h_action=args.h_action)

    # discretized state space (for plot purposes only)
    env.discretize_state_space(h_state=0.01)

    # run reinforce algorithm for stochastic policy and discrete actions
    data = reinforce(
        env,
        gamma=args.gamma,
        lr=args.lr,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    returns = data['returns']
    time_steps = data['time_steps']
    model = data['model']

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # compute action probability distributions
    states = np.expand_dims(env.state_space_h, axis=1)
    with torch.no_grad():
        action_prob_dists = model.forward(states).numpy()

    # sample policy
    policy = np.empty_like(env.state_space_h)
    for idx in range(env.n_states):
        idx_action = np.random.choice(np.arange(env.n_actions), p=action_prob_dists[idx])
        policy[idx] = np.array([env.action_space_h[idx_action]])

    plot_stoch_policy(env, action_prob_dists, policy, sol_hjb.u_opt)

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
