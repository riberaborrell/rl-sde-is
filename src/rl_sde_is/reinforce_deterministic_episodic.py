import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import FeedForwardNN
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def cumsum_torch(x):
    return torch.cumsum(x.flip(dims=(0,)), dim=0).flip(dims=(0,))

def get_total_reward(ep_rewards):
    psi = torch.empty_like(ep_rewards)
    psi[:] = torch.sum(ep_rewards)
    return psi

def get_reward_following_action(ep_rewards):
    return cumsum_torch(ep_rewards)

def get_reward_following_action_with_baseline(ep_rewards, ep_states, sol_hjb):
    idx = sol_hjb.sde.get_index_vectorized(ep_states.detach().unsqueeze(dim=1).numpy())[0]
    ep_baselines = torch.FloatTensor(sol_hjb.value_function[idx])
    return cumsum_torch(ep_rewards - ep_baselines)

def reinforce(env, return_estimator='total-rewards', gamma=0.99, n_layers=3, d_hidden_layer=30,
              batch_size=10, lr=0.01, n_episodes=2000, seed=1,
              sol_hjb=None, load=False, plot=False):

    # check type of the return estimator
    if return_estimator not in ['total-rewards', 'rewards-following-action', 'optimal-baseline']:
        raise ValueError

    # check ratio number of episodes and batch size
    if not n_episodes % batch_size == 0:
        raise ValueError

    # get dir path
    dir_path = get_reinforce_det_dir_path_old(
        env,
        agent='reinforce-det-episodic_' + return_estimator,
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

    # get dimensions of each layer
    d_hidden_layers = [d_hidden_layer for i in range(n_layers-1)]

    model = FeedForwardNN(d_in=env.state_space_dim, hidden_sizes=d_hidden_layers,
                          d_out=env.action_space_dim)

    # preallocate tensors for the tajectories in a batch
    batch_states = torch.empty((0,), dtype=torch.float32)
    batch_actions = torch.empty((0,), dtype=torch.float32)
    batch_rewards = torch.empty((0,), dtype=torch.float32)
    batch_brownian_increments = torch.empty((0,), dtype=torch.float32)
    batch_psi = torch.empty((0,), dtype=torch.float32)
    batch_counter = 0

    # preallocate lists to hold results
    losses = []
    var_losses = []
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

        # preallocate states and rewards for the episode
        ep_states = torch.empty((0,), dtype=torch.float32)
        ep_rewards = torch.empty((0,), dtype=torch.float32)

        # time step
        k = 0

        complete = False
        while complete == False:

            # save state
            ep_states = torch.cat((ep_states, state), 0)
            #batch_states = torch.cat((batch_states, state), 0)

            # get action following policy
            action = model.forward(state)

            # next step
            state, r, complete, dbt = env.step_torch(state, action)
            k += 1

            # save action, reward and brownian increment
            batch_actions = torch.cat((batch_actions, action), 0)
            batch_brownian_increments = torch.cat((batch_brownian_increments, dbt), 0)
            ep_rewards = torch.cat((ep_rewards, r), 0)

        # compute advantatge function
        if return_estimator == 'total-rewards' :
            ep_psi = get_total_reward(ep_rewards)
        elif return_estimator == 'rewards-following-action' :
            ep_psi = get_reward_following_action(ep_rewards)
        elif return_estimator == 'optimal-baseline' :
            ep_psi = get_reward_following_action_with_baseline(ep_rewards, ep_states, sol_hjb)

        # update batch data
        batch_states = torch.cat((batch_states, ep_states), 0)
        batch_rewards = torch.cat((batch_rewards, ep_rewards), 0)
        batch_psi = torch.cat((batch_psi, ep_psi), 0,)
        batch_counter += 1

        # save info
        returns.append(sum(ep_rewards.detach().numpy()))
        time_steps.append(k)

        # update network if batch is complete 
        if batch_counter == batch_size:

            # reset gradients ..
            optimizer.zero_grad()

            # tensor states, actions and rewards
            n_steps = batch_states.shape[0]

            # compute grad log probability transition density
            batch_log_probs = batch_actions * batch_brownian_increments

            # calculate loss
            loss = torch.sum(- batch_rewards - batch_psi * batch_log_probs) / batch_size

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # save info
            losses.append(loss.item())

            # reset batch
            batch_states = torch.empty((0,), dtype=torch.float32)
            batch_actions = torch.empty((0,), dtype=torch.float32)
            batch_rewards = torch.empty((0,), dtype=torch.float32)
            batch_brownian_increments = torch.empty((0,), dtype=torch.float32)
            batch_psi = torch.empty((0,), dtype=torch.float32)
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

    # discretized state space (for plot purposes only)
    env.discretize_state_space(h_state=0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run reinforce
    data = reinforce(
        env=env,
        return_estimator=args.return_estimator,
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr=args.lr,
        n_episodes=args.n_episodes,
        sol_hjb=sol_hjb,
        load=args.load,
        plot=args.plot,
    )

    # plots
    if not args.plot:
        return

    # plot moving averages for each episode
    returns = data['returns']
    run_mean_returns = compute_running_mean(returns, args.batch_size)
    run_var_returns = compute_running_variance(returns, args.batch_size)
    time_steps = data['time_steps']
    run_mean_time_steps = compute_running_mean(time_steps, args.batch_size)
    #plot_returns_episodes(returns, run_mean_returns)
    #plot_run_var_returns_episodes(run_var_returns)
    #plot_run_mean_returns_with_error_episodes(run_mean_returns, run_var_returns)
    #plot_time_steps_episodes(time_steps, run_mean_time_steps)

    # plot expected values for each epoch
    test_mean_returns = run_mean_returns[::args.batch_size]
    test_var_returns = run_var_returns[::args.batch_size]
    #test_mean_lengths = data['test_mean_lengths']
    plot_expected_returns_with_error_epochs(test_mean_returns, test_var_returns)

    # get model
    model = data['model']

    # compute actions following the deterministic policy
    policy = compute_det_policy_actions(env, model)

    # plot policy
    plot_det_policy(env, policy, sol_hjb.u_opt)


if __name__ == '__main__':
    main()
