import gymnasium as gym
import gym_sde_is
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.spg.spg_utils import *
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.numeric import cumsum_numpy as cumsum
from rl_sde_is.utils.path import get_reinforce_discrete_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import *


def reinforce(env, n_layers=2, d_hidden_layer=32, n_episodes=1000, batch_size=10, lr=1e-2,
              seed=None, h_action=0.1, load=False):

    # get dir path
    dir_path = get_reinforce_discrete_dir_path(
        env,
        agent='reinforce-stoch-discrete',
        h_action=h_action,
        batch_size=batch_size,
        lr=lr,
        n_episodes=n_episodes,
        seed=seed,
    )

    # discretize action space
    env.discretize_action_space(h_action=h_action)

    # load results
    if load:
        return load_data(dir_path)

    # number of gradient iterations
    n_iterations = n_episodes // batch_size

    # save algorithm parameters
    data = {
        'h_action': h_action,
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'n_episodes': n_episodes,
        'batch_size' : batch_size,
        'n_iterations': n_iterations,
        'lr' : lr,
        'seed': seed,
        'dir_path': dir_path,
    }

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initialize policy
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
    policy = CategoricalPolicy(
        state_dim=env.d, n_actions=env.n_actions, hidden_sizes=hidden_sizes, activation=nn.Tanh(),
    )

    # define optimizer
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # preallocate batch arrays 
    batch_states = np.empty((0, env.d), dtype=np.float32)
    batch_actions_idx = np.empty(0, dtype=np.int64)
    batch_n_returns = np.empty(0, dtype=np.float32)
    batch_returns = np.empty(0, dtype=np.float32)
    batch_time_steps = np.empty(0, dtype=np.int64)
    batch_counter = 0

    # preallocate iteration arrays
    objectives = np.empty(0, dtype=np.float32)
    losses = np.empty(0, dtype=np.float32)
    #loss_vars = np.empty(0, dtype=np.float32)
    mfhts = np.empty(0, dtype=np.float32)

    for ep in np.arange(n_episodes):

        # initialization
        state, _ = env.reset()

        # preallocate rewards for the episode
        ep_rewards = np.empty(0, dtype=np.float32)

        done = False
        while done == False:

            # save state
            batch_states = np.vstack([batch_states, state])

            # sample action
            action_idx, _ = policy.sample_action(torch.FloatTensor(state))
            action = env.action_space_h[action_idx]

            # next step
            state, r, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            # save action and reward
            batch_actions_idx = np.append(batch_actions_idx, action_idx)
            ep_rewards = np.append(ep_rewards, r)

        # update batch data
        ep_returns = cumsum(ep_rewards)
        batch_n_returns = np.append(batch_n_returns, ep_returns)
        batch_returns = np.append(batch_returns, ep_returns[0])
        batch_time_steps = np.append(batch_time_steps, env.n_steps)
        batch_counter += 1

        # update network if batch is complete 
        if batch_counter == batch_size:

            # reset gradients ..
            optimizer.zero_grad()

            # torchify arrays
            states = torch.FloatTensor(batch_states)
            idx_action = torch.LongTensor(batch_actions_idx)
            n_returns = torch.FloatTensor(batch_n_returns)

            #TODO: debug
            # calculate value function at each state
            #states_idx = sol_hjb.sde.get_index_vectorized(states.numpy())
            #values = torch.FloatTensor(-sol_hjb.value_function[states_idx])
            #values = values.unsqueeze(1)

            # calculate loss
            _, log_probs = policy.forward(states, idx_action)
            loss = - (n_returns * log_probs).sum() / batch_size
            #var_loss = (returns * log_probs).detach().numpy().var()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # save stats
            objectives = np.append(objectives, batch_returns.mean())
            losses = np.append(losses, loss.detach().numpy())
            #loss_vars = np.append(loss_vars, loss.detach().numpy())
            mfhts = np.append(mfhts, env.dt * batch_time_steps.mean())

            # reset batch
            batch_states = np.empty((0, env.d), dtype=np.float32)
            batch_actions_idx = np.empty(0, dtype=np.int64)
            batch_n_returns = np.empty(0, dtype=np.float32)
            batch_returns = np.empty(0, dtype=np.float32)
            batch_time_steps = np.empty(0, dtype=np.int64)
            batch_counter = 0

            # log
            print('ep: {}, objective: {:.2f}, loss: {:.2e}, mfht: {:.2e}' \
                  ''.format(ep, objectives[-1], losses[-1], mfhts[-1]))

    data['objectives'] = objectives
    data['losses'] = losses
    data['mfht'] = mfhts
    data['policy'] = policy
    save_data(data, dir_path)
    return data

def main():
    args = get_base_parser().parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        alpha=np.array(args.alpha),
        beta=args.beta,
        state_init_dist=args.state_init_dist,
    )

    # discretize state space
    h_coarse = 0.01
    env.discretize_state_space(h_coarse)

    # load hjb solution
    sol_hjb = env.get_hjb_solver()
    sol_hjb.coarse_solution(h_coarse)
    policy_opt = sol_hjb.u_opt

    # run reinforce
    data = reinforce(
        env,
        h_action=args.h_action,
        lr=args.lr,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        load=args.load,
    )

    # plot results
    if not args.plot:
        return

    # plot returns and time steps
    n_iterations = data['n_iterations']
    objectives, mfhts = data['objectives'], data['mfht']
    x = np.arange(n_iterations)
    plot_y_per_x(x, objectives, title='Losses', xlabel='Iterations')
    plot_y_per_x(x, mfhts, title='MFHT', xlabel='Iterations')

    # sampled actions following learnt policy
    policy = data['policy']
    actions_idx, _ = policy.sample_action(torch.FloatTensor(env.state_space_h))
    policy_h = env.action_space_h[actions_idx]

    # action probabiliy distributions
    with torch.no_grad():
        dists, _ = policy.forward(torch.FloatTensor(env.state_space_h))
        action_probs = dists.probs.numpy()

    # plot stochastic policy
    plot_categorical_stoch_policy_1d(env, action_probs, policy_h, policy_opt)


if __name__ == '__main__':
    main()
