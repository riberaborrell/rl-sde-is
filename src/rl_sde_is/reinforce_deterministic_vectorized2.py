import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import mlp
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *
from rl_sde_is.approximate_methods import *

def get_parser():
    parser = get_base_parser()
    return parser

class DeterministicPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.policy = mlp(sizes=sizes, activation=activation, output_activation=nn.Identity)

    def forward(self, state):
        return self.policy.forward(state)

def sample_loss_vectorized(env, model, K, control_hjb=None):

    # max time steps
    dt = env.dt_tensor
    k_max = 10**6

    # start timer
    ct_initial = time.time()

    # initialize trajectories
    states = torch.FloatTensor(env.reset_vectorized(batch_size=K))

    # initialize running integral
    return_t = torch.zeros(K)
    return_fht = torch.empty(K)
    stoch_int_t = torch.zeros(K)
    stoch_int_fht = torch.empty(K)

    # preallocate time steps
    time_steps = np.empty(K)

    # preallocate hjb policy l2 error
    policy_l2_error_fht = np.empty(K)
    policy_l2_error_fht.fill(np.nan)
    if control_hjb is not None:
        policy_l2_error_t = np.zeros(K)

    # are episode finish?
    already_done = torch.full((K, 1), False)
    done = torch.full((K, 1), False)

    for k in np.arange(1, k_max + 1):

        # actions
        actions = model.forward(states)

        # step dynamics forward
        next_states, rewards, done, dbt = env.step_vectorized_torch(states, actions)

        # update work with running cost
        return_t = return_t + rewards.squeeze()

        # update stochastic integral
        stoch_int_t = stoch_int_t + torch.matmul(
            actions[:, np.newaxis, :],
            dbt[:, :, np.newaxis],
        ).squeeze()

        # computer running u l2 error
        if control_hjb is not None:

            # hjb control
            idx_states = env.get_states_idx_vectorized(states.detach().numpy())
            actions_hjb = control_hjb[idx_states]

            # update running u l2 error
            policy_l2_error_t += (
                np.linalg.norm(actions.detach().numpy() - actions_hjb, axis=1) ** 2
            ) * env.dt

        # get indices of trajectories which are new to the target set
        idx = env.get_idx_new_in_ts_torch(done, already_done)

        if idx.shape[0] != 0:

            # fix work running integral
            return_fht[idx] = return_t.index_select(0, idx)

            # fix running integrals
            stoch_int_fht[idx] = stoch_int_t.index_select(0, idx)

            # time steps
            time_steps[idx] = k

            # fix policy l2 error
            if control_hjb is not None:
                policy_l2_error_fht[idx] = policy_l2_error_t[idx]

        # stop if xt_traj in target set
        if already_done.all() == True:
           break

        # update states
        states = next_states

    # compute effective loss
    eff_loss = torch.mean(-return_fht - return_fht.detach() * stoch_int_fht)

    # end timer
    ct_final = time.time()

    return eff_loss, return_fht.detach().numpy(), time_steps, \
           policy_l2_error_fht.mean(), ct_final - ct_initial

def reinforce(env, gamma=0.99, n_layers=3, d_hidden_layer=256,
              batch_size=1000, lr=1e-3, n_iterations=100, backup_freq_iterations=None, seed=None,
              control_hjb=None, load=False, plot=False):

    # get dir path
    rel_dir_path = get_reinforce_det_dir_path(
        env,
        agent='reinforce-deterministic',
        d_hidden_layer=d_hidden_layer,
        batch_size=batch_size,
        lr=lr,
        n_iterations=n_iterations,
        seed=seed,
    )

    # load results
    if load:
        data = load_data(rel_dir_path)
        return data

    # set seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # get dimensions of each layer
    d_hidden_layers = [d_hidden_layer for i in range(n_layers-1)]

    # initialize nn model 
    model = DeterministicPolicy(state_dim=env.state_space_dim, action_dim=env.action_space_dim,
                                hidden_sizes=d_hidden_layers, activation=nn.Tanh)

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # preallocate arrays providing information at each gradient step
    losses = np.empty(n_iterations)
    exp_returns = np.empty(n_iterations)
    var_returns = np.empty(n_iterations)
    exp_time_steps = np.empty(n_iterations)
    policy_l2_errors = np.empty(n_iterations)
    cts = np.empty(n_iterations)
    losses.fill(np.nan)
    exp_returns.fill(np.nan)
    var_returns.fill(np.nan)
    exp_time_steps.fill(np.nan)
    policy_l2_errors.fill(np.nan)
    cts.fill(np.nan)

    # preallocate list of returns and time steps
    returns = np.empty(0, dtype=np.float32)
    time_steps = np.empty(0, dtype=np.int32)

    # save algorithm parameters
    data = {
        'gamma': gamma,
        'd_hidden_layer': d_hidden_layer,
        'batch_size': batch_size,
        'lr': lr,
        'n_iterations': n_iterations,
        'seed': seed,
        'backup_freq_iterations': backup_freq_iterations,
        'model': model,
        'rel_dir_path': rel_dir_path,
    }
    save_data(data, rel_dir_path)

    # save model initial parameters
    save_model(model, rel_dir_path, 'model_n-it{}'.format(0))

    # initialize animated figures
    if plot:
        state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
        with torch.no_grad():
            initial_policy = model.forward(state_space_h).numpy().squeeze()
        policy_line = initialize_det_policy_figure(env, initial_policy, control_hjb)

    for i in np.arange(n_iterations):

        # reset gradients
        optimizer.zero_grad()

        # compute effective loss
        eff_loss, batch_returns, batch_time_steps, policy_l2_error, ct \
                = sample_loss_vectorized(env, model, batch_size, control_hjb)
        eff_loss.backward()

        # update parameters
        optimizer.step()

        # save statistics
        returns = np.append(returns, batch_returns)
        time_steps = np.append(time_steps, batch_time_steps)
        losses[i] = eff_loss.detach().numpy()
        exp_returns[i] = np.mean(batch_returns)
        var_returns[i] = np.var(batch_returns)
        exp_time_steps[i] = np.mean(batch_time_steps)
        policy_l2_errors[i] = policy_l2_error
        cts[i] = ct

        msg = 'it.: {:2d}, loss: {:.3e}, exp return: {:.3e}, var return: {:.1e}, ' \
              'avg ts: {:.3e}, policy l2-error: {:.2e}, ct: {:.3f}' \
              ''.format(
                  i,
                  losses[i],
                  exp_returns[i],
                  var_returns[i],
                  exp_time_steps[i],
                  policy_l2_error,
                  ct,
              )
        print(msg)

        # backupa models and results
        if backup_freq_iterations is not None and (i + 1) % backup_freq_iterations == 0:

            # save model
            save_model(model, rel_dir_path, 'model_n-it{}'.format(i + 1))

            # add results
            data['returns'] = returns
            data['time_steps'] = time_steps
            data['losses'] = losses
            data['exp_returns'] = exp_returns
            data['var_returns'] = var_returns
            data['exp_time_steps'] = exp_time_steps
            data['policy_l2_errors'] = policy_l2_errors
            data['cts'] = cts
            save_data(data, rel_dir_path)

        # update figure
        if plot and i % 1 == 0:
            state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
            with torch.no_grad():
                policy = model.forward(state_space_h).numpy().squeeze()
            update_det_policy_figure(env, policy, policy_line)

    # add results
    data['returns'] = returns
    data['time_steps'] = time_steps
    data['losses'] = losses
    data['exp_returns'] = exp_returns
    data['var_returns'] = var_returns
    data['exp_time_steps'] = exp_time_steps
    data['policy_l2_errors'] = policy_l2_errors
    data['cts'] = cts
    save_data(data, rel_dir_path)
    return data

def load_backup_model(data, it=0):
    try:
        load_model(data['model'], data['rel_dir_path'], file_name='model_n-it{}'.format(it))
    except FileNotFoundError as e:
        print('there is no backup for iteration {:d}'.format(it))

def get_policy(env, data, it=None):
    model = data['model']
    if it is not None:
        load_backup_model(data, it)

    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    with torch.no_grad():
        policy = model.forward(state_space_h).numpy().squeeze()
    return policy

def get_policies(env, data, iterations):

    Nx = env.n_states
    policies = np.empty((0, Nx), dtype=np.float32)

    for it in iterations:
        load_backup_model(data, it)
        policies = np.vstack((policies, get_policy(env, data).reshape(1, Nx)))

    return policies

def get_backup_policies(env, data):
    n_iterations = data['n_iterations']
    backup_freq_iterations = data['backup_freq_iterations']

    Nx = env.n_states
    policies = np.empty((0, Nx), dtype=np.float32)

    for i in range(data['n_iterations']):
        if i == 0:
            load_backup_model(data, 0)
            policies = np.vstack((policies, get_policy(env, data).reshape(1, Nx)))

        #elif (i + 1) % backup_freq_iterations == 0:
        elif (i + 1) % 100 == 0:
            load_backup_model(data, i+1)
            policies = np.vstack((policies, get_policy(env, data).reshape(1, Nx)))

    return policies


def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta)

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # set action space bounds
    env.action_space_low = 0
    env.action_space_high = 5

    # discretized state space (for plot purposes only)
    env.discretize_state_space(h_state=0.05)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run reinforve algorithm with a deterministic policy
    data = reinforce(
        env,
        gamma=args.gamma,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=args.n_iterations,
        backup_freq_iterations=args.backup_freq_iterations,
        seed=args.seed,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )

    # do plots
    if not args.plot:
        return

    # plot moving averages for each episode
    returns = data['returns']
    run_mean_returns = compute_running_mean(returns, args.batch_size)
    run_var_returns = compute_running_variance(returns, args.batch_size)
    time_steps = data['time_steps']
    run_mean_time_steps = compute_running_mean(time_steps, args.batch_size)
    plot_returns_episodes(returns, run_mean_returns)
    #plot_run_var_returns_episodes(run_var_returns)
    #plot_run_mean_returns_with_error_episodes(run_mean_returns, run_var_returns)
    plot_time_steps_episodes(time_steps, run_mean_time_steps)
    return

    # plot policy
    policy = get_policy(env, data, it=args.plot_iteration)
    plot_det_policy(env, policy, sol_hjb.u_opt)

    #iterations = np.linspace(0, 4000, 6, dtype=np.int32)
    #policies = get_policies(env, data, iterations)
    policies = get_backup_policies(env, data)
    plot_det_policies(env, policies, sol_hjb.u_opt)
    #plot_det_policies_black_and_white(env, policies, sol_hjb.u_opt)

    # plot expected values for each epoch
    plot_expected_returns_with_error_epochs(data['exp_returns'], data['var_returns'])
    plot_time_steps_epochs(data['exp_time_steps'])

    # plot policy l2 error
    plot_det_policy_l2_error_epochs(data['policy_l2_errors'])

    # plot loss function
    plot_loss_epochs(data['losses'])


if __name__ == "__main__":
    main()
