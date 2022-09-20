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

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def cumsum_torch(x):
    return torch.cumsum(x.flip(dims=(0,)), dim=0).flip(dims=(0,))

def discount_cumsum_torch(x, gamma):
    import scipy
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
    input:
        vector x,
        [x0,
         x1,
         x2]

     output:
        [x0 + gamma * x1 + gamma^2 * x2,
         x1 + gamma * x2,
         x2]
    """
    breakpoint()
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]

def normalize_advs_trick(x):
    return (x - np.mean(x))/(np.std(x) + 1e-8)

def get_total_reward(ep_rewards):
    psi = torch.empty_like(ep_rewards)
    psi[:] = torch.sum(ep_rewards)
    return psi

def get_reward_following_action(ep_rewards):
    return cumsum_torch(ep_rewards)

def get_reward_following_action_with_baseline(ep_rewards, ep_states, value_function):
    pass

def reinforce(env, gamma=0.99, n_layers=3, d_hidden_layer=30,
              batch_size=10, lr=0.01, n_episodes=2000, seed=1,
              value_function_hjb=None, control_hjb=None, load=False, plot=False):

    assert n_episodes % batch_size == 0, ''
    n_iterations = int(n_episodes / batch_size)

    # get dir path
    dir_path = get_reinforce_det_dir_path(
        env,
        agent='reinforce-det-episodic',
        batch_size=batch_size,
        lr=lr,
        n_iterations=n_iterations,
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

    # preallocate lists to hold results
    batch_states = torch.empty((0,), dtype=torch.float32)
    batch_actions = torch.empty((0,), dtype=torch.float32)
    batch_rewards = torch.empty((0,), dtype=torch.float32)
    batch_brownian_increments = torch.empty((0,), dtype=torch.float32)
    batch_psi = torch.empty((0,), dtype=torch.float32)
    #batch_det_int_fht = torch.empty((0,), dtype=torch.float32)
    #batch_stoch_int_fht = torch.empty((0,), dtype=torch.float32)
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
        ep_rewards = torch.empty((0,), dtype=torch.float32)

        # running deterministic and stochastic integrals
        #det_int_t = torch.zeros(1)
        #stoch_int_t = torch.zeros(1)

        # time step
        k = 0

        complete = False
        while complete == False:

            # save state
            batch_states = torch.cat((batch_states, state), 0)

            # get action following policy
            action = model.forward(state)

            # next step
            state, r, complete, dbt = env.step_torch(state, action)
            k += 1

            # save action, reward and brownian increment
            batch_actions = torch.cat((batch_actions, action), 0)
            batch_brownian_increments = torch.cat((batch_brownian_increments, dbt), 0)
            ep_rewards = torch.cat((ep_rewards, r), 0)

            # update deterministic integral
            #det_int_t += (torch.linalg.norm(action) ** 2) * env.dt_tensor

            # update stochastic integral
            # compute grad log probability transition function
            #stoch_int_t += torch.dot(action, dbt)

        # update batch data
        batch_rewards = torch.cat((batch_rewards, ep_rewards), 0)
        #ep_psi = get_total_reward(ep_rewards)
        ep_psi = get_reward_following_action(ep_rewards)
        batch_psi = torch.cat((batch_psi, ep_psi), 0,)
        #batch_det_int_fht = torch.cat((batch_det_int_fht, det_int_t), 0)
        #batch_stoch_int_fht = torch.cat((batch_stoch_int_fht, stoch_int_t), 0)
        batch_counter += 1
        returns.append(sum(ep_rewards.detach().numpy()))
        time_steps.append(k)

        # update network if batch is complete 
        if batch_counter == batch_size:

            # reset gradients ..
            optimizer.zero_grad()

            # tensor states, actions and rewards
            n_steps = batch_states.shape[0]

            # compute gradient of the return
            #batch_grad_return = 0.5 * (batch_actions ** 2) * env.dt_tensor

            # compute grad log probability transition density
            batch_log_probs = batch_actions * batch_brownian_increments

            # calculate loss
            #loss = - (batch_discounted_returns * grad_log_probs).mean()
            loss = torch.sum(batch_rewards + batch_psi * batch_log_probs) / batch_size

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # reset batch
            batch_states = torch.empty((0,), dtype=torch.float32)
            batch_actions = torch.empty((0,), dtype=torch.float32)
            batch_rewards = torch.empty((0,), dtype=torch.float32)
            batch_brownian_increments = torch.empty((0,), dtype=torch.float32)
            batch_psi = torch.empty((0,), dtype=torch.float32)
            #batch_det_int_fht = torch.empty((0,), dtype=torch.float32)
            #batch_stoch_int_fht = torch.empty((0,), dtype=torch.float32)
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
        #'controls': controls,
        #'losses': losses,
        #'var_losses': var_losses,
        #'means_I_u': means_I_u,
        #'vars_I_u': vars_I_u,
        #'res_I_u': res_I_u,
        #'cts': cts,
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
        gamma=args.gamma,
        batch_size=args.batch_size,
        lr=args.lr,
        n_episodes=args.n_episodes,
        value_function_hjb=sol_hjb.value_function,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )
    returns = data['returns']
    time_steps = data['time_steps']
    model = data['model']

    # do plots
    if args.plot:

        # smoothed arrays
        avg_returns = compute_smoothed_array(returns, args.batch_size)
        avg_time_steps = compute_smoothed_array(time_steps, args.batch_size)

        # compute actions following the deterministic policy
        policy = compute_det_policy_actions(env, model)

        plot_returns_episodes(returns, avg_returns)
        plot_time_steps_episodes(time_steps, avg_time_steps)
        plot_det_policy(env, policy, sol_hjb.u_opt)

if __name__ == '__main__':
    main()
