import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from rl_sde_is.approximate_methods import *
from rl_sde_is.models import mlp
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *

class DeterministicPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_limit):
        super().__init__()
        self.sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.policy = mlp(sizes=self.sizes, activation=activation)
        self.action_limit = action_limit
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-3, 5e-3)
                nn.init.uniform_(module.bias, -5e-3, 5e-3)

    def forward(self, state):
        return self.policy.forward(state)
        #action = self.policy.forward(state)
        #return self.action_limit * torch.tanh(action)

def sample_loss_vectorized(env, model, K):

    # max time steps
    dt = env.dt_tensor
    k_max = 10**6

    # start timer
    ct_initial = time.time()

    # initialize trajectories
    states = torch.FloatTensor(env.reset(batch_size=K))

    # initialize running integral
    return_t = torch.zeros(K)
    return_fht = torch.empty(K)
    stoch_int_t = torch.zeros(K)
    stoch_int_fht = torch.empty(K)

    # preallocate time steps
    time_steps = np.empty(K)

    # are episodes done
    already_done = torch.full((K,), False)
    done = torch.full((K,), False)

    for k in np.arange(1, k_max + 1):

        # actions
        actions = model.forward(states)

        # step dynamics forward
        next_states, rewards, done, dbt = env.step_torch(states, actions)

        # update work with running cost
        return_t = return_t + rewards.squeeze()

        # update stochastic integral
        stoch_int_t = stoch_int_t + torch.matmul(
            actions[:, np.newaxis, :],
            dbt[:, :, np.newaxis],
        ).squeeze()

        # get indices of trajectories which are new to the target set
        idx = env.get_idx_new_in_ts_torch(done, already_done)

        if idx.shape[0] != 0:

            # fix work running integral
            return_fht[idx] = return_t.index_select(0, idx)

            # fix running integrals
            stoch_int_fht[idx] = stoch_int_t.index_select(0, idx)

            # time steps
            time_steps[idx] = k

        # stop if xt_traj in target set
        if already_done.all() == True:
           break

        # update states
        states = next_states

    # compute effective loss
    eff_loss = torch.mean(-return_fht - return_fht.detach() * stoch_int_fht)

    # end timer
    ct_final = time.time()

    return eff_loss, return_fht.detach().numpy(), time_steps, ct_final - ct_initial

def reinforce(env, gamma=0.99, d_hidden_layer=256, n_layers=3, action_limit=5,
              batch_size=1000, lr=1e-3, n_iterations=100, test_batch_size=1000,
              test_freq_iterations=100, backup_freq_iterations=None, seed=None,
              policy_opt=None, load=False, live_plot=False):

    # get dir path
    rel_dir_path = get_reinforce_det_dir_path(
        env,
        agent='reinforce-deterministic',
        gamma=gamma,
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
    model = DeterministicPolicy(
        state_dim=env.state_space_dim,
        action_dim=env.action_space_dim,
        hidden_sizes=d_hidden_layers,
        activation=nn.Tanh(),
        action_limit=action_limit,
    )

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # save algorithm parameters
    data = {
        'gamma': gamma,
        'd_hidden_layer': d_hidden_layer,
        'batch_size': batch_size,
        'lr': lr,
        'n_iterations': n_iterations,
        'seed': seed,
        'backup_freq_iterations': backup_freq_iterations,
        'test_freq_iterations': test_freq_iterations,
        'test_batch_size': test_batch_size,
        'model': model,
        'rel_dir_path': rel_dir_path,
    }
    save_data(data, rel_dir_path)

    # save model initial parameters
    save_model(model, rel_dir_path, 'model_n-it{}'.format(0))


    # preallocate arrays

    # returns and time steps for each episode
    returns = np.empty(0, dtype=np.float32)
    time_steps = np.empty(0, dtype=np.int32)

    # losses, expected returns, variances of returns and ct for each gradient step
    losses = np.empty(n_iterations)
    exp_returns = np.empty(n_iterations)
    var_returns = np.empty(n_iterations)
    cts = np.empty(n_iterations)
    losses.fill(np.nan)
    exp_returns.fill(np.nan)
    var_returns.fill(np.nan)
    cts.fill(np.nan)

    # test mean, variance and mean length of the returns and l2 error after each epoch
    test_mean_returns = np.empty((0), dtype=np.float32)
    test_var_returns = np.empty((0), dtype=np.float32)
    test_mean_lengths = np.empty((0), dtype=np.float32)
    test_policy_l2_errors = np.empty((0), dtype=np.float32)

    # test initial model
    test_mean_ret, test_var_ret, test_mean_len, test_policy_l2_error \
            = test_policy_vectorized(env, model, batch_size=test_batch_size,
                                     policy_opt=policy_opt)
    test_mean_returns = np.append(test_mean_returns, test_mean_ret)
    test_var_returns = np.append(test_var_returns, test_var_ret)
    test_mean_lengths = np.append(test_mean_lengths, test_mean_len)
    test_policy_l2_errors = np.append(test_policy_l2_errors, test_policy_l2_error)

    msg = 'it.: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
          'test mean time steps: {:2.2f}, test policy l2 error: {:.2e}'.format(
              0,
              test_mean_ret,
              test_var_ret,
              test_mean_len,
              test_policy_l2_error,
          )
    print(msg)

    # initialize live figures
    if live_plot and env.d == 1:
        policy_line = initialize_1d_figures(env, model, policy_opt)
    elif live_plot and env.d == 2:
        Q_policy = initialize_2d_figures(env, model, policy_opt)

    for i in np.arange(n_iterations):

        # reset gradients
        optimizer.zero_grad()

        # compute effective loss
        eff_loss, batch_returns, batch_time_steps, ct \
                = sample_loss_vectorized(env, model, batch_size)
        eff_loss.backward()

        # update parameters
        optimizer.step()

        # save statistics
        returns = np.append(returns, batch_returns)
        time_steps = np.append(time_steps, batch_time_steps)
        losses[i] = eff_loss.detach().numpy()
        exp_returns[i] = np.mean(batch_returns)
        var_returns[i] = np.var(batch_returns)
        cts[i] = ct

        msg = 'it.: {:2d}, loss: {:.3e}, exp return: {:.3e}, var return: {:.1e}, ' \
              'ct: {:.3f}' \
              ''.format(
                  i,
                  losses[i],
                  exp_returns[i],
                  var_returns[i],
                  ct,
              )
        print(msg)

        # test model
        if (i + 1) % test_freq_iterations == 0:

            test_mean_ret, test_var_ret, test_mean_len, test_policy_l2_error \
                    = test_policy_vectorized(env, model, batch_size=test_batch_size,
                                             policy_opt=policy_opt)
            test_mean_returns = np.append(test_mean_returns, test_mean_ret)
            test_var_returns = np.append(test_var_returns, test_var_ret)
            test_mean_lengths = np.append(test_mean_lengths, test_mean_len)
            test_policy_l2_errors = np.append(test_policy_l2_errors, test_policy_l2_error)

            msg = 'it.: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
                  'test mean time steps: {:2.2f}, test policy l2 error: {:.2e}'.format(
                i + 1,
                test_mean_ret,
                test_var_ret,
                test_mean_len,
                test_policy_l2_error,
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
            data['cts'] = cts
            data['test_mean_returns'] = test_mean_returns
            data['test_var_returns'] = test_var_returns
            data['test_mean_lengths'] = test_mean_lengths
            data['test_policy_l2_errors'] = test_policy_l2_errors
            save_data(data, rel_dir_path)

        # update figure
        if live_plot and i % 1 == 0:

            if env.d == 1:
                update_1d_figures(env, model, policy_line)
            elif env.d == 2:
                update_2d_figures(env, model, Q_policy)

    # add results
    data['returns'] = returns
    data['time_steps'] = time_steps
    data['losses'] = losses
    data['exp_returns'] = exp_returns
    data['var_returns'] = var_returns
    data['cts'] = cts
    data['test_mean_returns'] = test_mean_returns
    data['test_var_returns'] = test_var_returns
    data['test_mean_lengths'] = test_mean_lengths
    data['test_policy_l2_errors'] = test_policy_l2_errors
    save_data(data, rel_dir_path)
    return data

def initialize_1d_figures(env, model, policy_opt):

    # hjb control
    if policy_opt is None:
        policy_opt_plot = np.empty_like(env.state_space_h)
        policy_opt_plot.fill(np.nan)
    else:
        policy_opt_plot = policy_opt

    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    initial_policy = compute_det_policy_actions(env, model, state_space_h).squeeze()
    policy_line = initialize_det_policy_1d_figure(env, initial_policy, policy_opt_plot)

    return policy_line

def update_1d_figures(env, model, policy_line):
    states = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    policy = compute_det_policy_actions(env, model, states)
    update_det_policy_1d_figure(env, policy, policy_line)

def initialize_2d_figures(env, model, policy_hjb):
    states = torch.FloatTensor(env.state_space_h)
    initial_policy = compute_det_policy_actions(env, model, states)
    Q_policy = initialize_det_policy_2d_figure(env, initial_policy, policy_hjb)
    return Q_policy

def update_2d_figures(env, model, Q_policy):
    states = torch.FloatTensor(env.state_space_h)
    policy = compute_det_policy_actions(env, model, states)
    update_det_policy_2d_figure(env, policy, Q_policy)
