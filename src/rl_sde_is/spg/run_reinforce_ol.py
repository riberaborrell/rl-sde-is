import gymnasium as gym
import gym_sde_is
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.spg.spg_utils import *
from rl_sde_is.approximate_methods import *
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.numeric import cumsum_numpy as cumsum
from rl_sde_is.utils.path import get_reinforce_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import *

def reinforce(env, n_layers=2, d_hidden_layer=32, policy_cov='const', expl_noise_init=1., lr=1e-4,
              n_episodes=1000, batch_size=1, seed=None, backup_freq=None,
              value_function_opt=None, policy_opt=None, load=False, live_plot=False):

    # get dir path
    dir_path = get_reinforce_dir_path(
        env,
        agent='reinforce',
        policy_cov=policy_cov,
        batch_size=batch_size,
        lr=lr,
        n_episodes=n_episodes,
        seed=seed,
    )

    # load results
    if load:
        return load_data(dir_path)

    # number of gradient iterations
    n_iterations = n_episodes // batch_size

    # save algorithm parameters
    data = {
        'n_layers': n_layers,
        'd_hidden_layer': d_hidden_layer,
        'policy_cov': policy_cov,
        'n_episodes': n_episodes,
        'batch_size' : batch_size,
        'n_iterations': n_iterations,
        'lr' : lr,
        'seed': seed,
        'dir_path': dir_path,
    }
    save_data(data, dir_path)

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # initialize model and optimizer
    hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]

    if policy_cov == 'const':
        policy = GaussianPolicyConstantCov(state_dim=env.d, action_dim=env.d,
                                           hidden_sizes=hidden_sizes, activation=nn.Tanh(),
                                           sigma=expl_noise_init)
    else:
        policy = GaussianPolicyLearntCov(
            state_dim=env.d, action_dim=env.d, hidden_sizes=hidden_sizes, activation=nn.Tanh(),
        )
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # preallocate batch arrays
    batch_states = np.empty((0, env.d), dtype=np.float32)
    batch_actions = np.empty((0, env.d), dtype=np.float32)
    batch_n_returns = np.empty(0, dtype=np.float32)
    batch_returns = np.empty(0, dtype=np.float32)
    batch_time_steps = np.empty(0, dtype=np.int64)
    batch_counter = 0

    # preallocate iteration arrays
    objectives = np.empty(0, dtype=np.float32)
    losses = np.empty(0, dtype=np.float32)
    #loss_vars = np.empty(0, dtype=np.float32)
    mfhts = np.empty(0, dtype=np.float32)

    if live_plot and env.d == 1:
        mean, sigma = compute_table_stoch_policy_1d(env, policy)
        lines = initialize_gaussian_policy_1d_figure(env, mean, sigma, policy_opt=policy_opt)

    for ep in np.arange(n_episodes):


        # initialization
        state, _ = env.reset()

        # preallocate episode rewards
        ep_rewards = np.empty(0, dtype=np.float32)

        # terminal state flag
        done = False
        while done == False:

            # save state
            batch_states = np.vstack([batch_states, state])

            # sample action
            action, _ = policy.sample_action(torch.FloatTensor(state))

            # env step
            state, r, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            # save action and reward
            batch_actions = np.vstack([batch_actions, action])
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
            actions = torch.FloatTensor(batch_actions)
            n_returns = torch.FloatTensor(batch_n_returns)

            # calculate loss
            _, log_probs = policy.forward(states, actions)
            breakpoint()
            loss = - (log_probs * n_returns).sum() / batch_size
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
            batch_actions = np.empty((0, env.d), dtype=np.float32)
            batch_n_returns = np.empty(0, dtype=np.float32)
            batch_returns = np.empty(0, dtype=np.float32)
            batch_time_steps = np.empty(0, dtype=np.int64)
            batch_counter = 0

            # log
            print('ep: {}, objective: {:.2f}, loss: {:.2e}, mfht: {:.2e}' \
                  ''.format(ep, objectives[-1], losses[-1], mfhts[-1]))

        # update plots
        if live_plot and env.d == 1 and (ep + 1) % 10 == 0:
            mean, sigma = compute_table_stoch_policy_1d(env, policy)
            update_gaussian_policy_1d_figure(env, mean, sigma, lines)

    data['objectives'] = objectives
    data['losses'] = losses
    data['mfht'] = mfhts
    data['policy'] = policy
    save_data(data, dir_path)
    return data

def main():
    parser = get_base_parser()
    parser.description = 'Run reinforce with fht for the sde importance sampling environment'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        alpha=np.array(args.alpha),
        beta=args.beta,
        state_init_dist=args.state_init_dist,
    )

    # discretize state and action space (plot purposes only)
    h_coarse = 0.1
    env.discretize_state_space(h_state=h_coarse)
    env.discretize_action_space(h_action=h_coarse)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    sol_hjb.coarse_solution(h_coarse)
    policy_opt = sol_hjb.u_opt
    value_function_opt = -sol_hjb.value_function

    # run simple dpg with known q-value function
    data = reinforce(
        env,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        policy_cov=args.policy_cov,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        n_episodes=args.n_episodes,
        backup_freq=args.backup_freq,
        value_function_opt=value_function_opt,
        policy_opt=policy_opt,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plot results
    if not args.plot:
        return

    # plot avg returns and mfht
    n_iterations = data['n_iterations']
    objectives, mfhts = data['objectives'], data['mfht']
    x = np.arange(n_iterations)
    plot_y_per_x(x, objectives, title='Losses', xlabel='Iterations')
    plot_y_per_x(x, mfhts, title='MFHT', xlabel='Iterations')

    # plot policy
    if env.d == 1:
        mean, sigma = compute_table_stoch_policy_1d(env, data['policy'])
        plot_gaussian_stoch_policy_1d(env, mean.squeeze(), sigma.squeeze(), policy_opt)

if __name__ == '__main__':
    main()
