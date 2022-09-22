import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.policies import GaussStochPolicy1, GaussStochPolicy2
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *

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
              batch_size=10, seed=1., load=False, plot=False):

    # get dir path
    dir_path = get_reinforce_stoch_dir_path(
        env,
        agent='reinforce-stochastic-continuous',
        batch_size=batch_size,
        lr=lr,
        n_episodes=n_episodes,
        seed=seed,
    )

    # load results
    if load:
        data = load_data(dir_path)
        return data

    # set seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # initialize policy
    model = GaussStochPolicy2(d_in=env.state_space_dim, hidden_size=32,
                              d_out=env.action_space_dim, seed=seed)

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # preallocate lists to hold results
    batch_states = []
    batch_actions = []
    batch_discounted_returns = []
    batch_counter = 0
    returns = []
    time_steps = []
    losses = []
    var_losses = []

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
            loss = (- discounted_returns_tensor * log_probs[:, 0]).mean()
            #loss = torch.sum(- discounted_returns_tensor * log_probs[:, 0]) / batch_size
            var_loss = (- discounted_returns_tensor * log_probs[:, 0]).detach().var()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # save loss and variance
            losses.append(loss.detach().numpy().item())
            var_losses.append(var_loss.numpy().item())

            # reset batch
            batch_states = []
            batch_actions = []
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
                )
            )

    data = {
        'n_episodes': n_episodes,
        'batch_size': batch_size,
        'returns': returns,
        'time_steps': time_steps,
        'losses': losses,
        'var_losses': var_losses,
        'model': model,
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

    # discretized state and action space (for plot purposes only)
    env.discretize_state_space(h_state=0.05)
    env.discretize_action_space(h_action=0.05)

    # run reinforce (stochastic policy gradient)
    data = reinforce(
        env=env,
        gamma=args.gamma,
        lr=args.lr,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        load=args.load,
        plot=args.plot,
    )

    # plots
    if not args.plot:
        return

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # plot moving averages for each episode
    returns = data['returns']
    run_mean_returns = compute_running_mean(returns, args.batch_size)
    run_var_returns = compute_running_variance(returns, args.batch_size)
    time_steps = data['time_steps']
    run_mean_time_steps = compute_running_mean(time_steps, args.batch_size)
    plot_run_mean_returns_with_error_episodes(run_mean_returns, run_var_returns)

    # plot losses
    losses = data['losses']
    var_losses = data['var_losses']
    plot_losses_with_errors_epochs(losses, var_losses)

    # get actions sampled following policy
    model = data['model']
    states = np.expand_dims(env.state_space_h, axis=1)
    policy = model.sample_action(states)[:, 0]

    # get action probability distributions
    actions = np.expand_dims(env.action_space_h, axis=1)
    action_prob_dists = np.empty((env.n_states, env.n_actions))
    #for idx in range(env.n_states):
    #    action_prob_dists[idx, :] = model.probability(states[idx], actions)[:, 0].detach().numpy()
    plot_stoch_policy(env, action_prob_dists, policy, sol_hjb.u_opt)

    # plot mu and sigma from gaussian stochastic policy
    with torch.no_grad():
        mu, sigma_sq = model.forward(states)
    mu = mu.numpy().squeeze()
    sigma_sq = sigma_sq.numpy().squeeze()
    plot_mu_and_simga_gaussian_stoch_policy(env, mu, sigma_sq)


if __name__ == '__main__':
    main()
