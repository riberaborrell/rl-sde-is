import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import FeedForwardNN, DenseNN
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    return parser

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

    return eff_loss, return_fht.detach().numpy(), np.mean(time_steps), \
           policy_l2_error_fht.mean(), ct_final - ct_initial

def reinforce(env, gamma=1.0, n_layers=3, d_hidden_layer=30, is_dense=False,
              batch_size=10, lr=0.01, n_iterations=100, backup_freq_iterations=None, seed=None,
              control_hjb=None, load=False, plot=False):

    # get dir path
    rel_dir_path = get_reinforce_det_dir_path(
        env,
        agent='reinforce-deterministic',
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
    model = FeedForwardNN(d_in=env.state_space_dim, hidden_sizes=d_hidden_layers,
                          d_out=env.action_space_dim, activation=nn.Tanh())

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # preallocate arrays
    losses = np.empty(n_iterations)
    exp_returns = np.empty(n_iterations)
    var_returns = np.empty(n_iterations)
    exp_time_steps = np.empty(n_iterations)
    policy_l2_errors = np.empty(n_iterations)
    cts = np.empty(n_iterations)

    # save initial parameters
    save_model(model, rel_dir_path, 'model_n-it{}'.format(0))

    for i in np.arange(n_iterations):

        # reset gradients
        optimizer.zero_grad()

        # compute effective loss
        eff_loss, return_fht, avg_len, policy_l2_error, ct \
                = sample_loss_vectorized(env, model, batch_size, control_hjb)
        eff_loss.backward()

        # update parameters
        optimizer.step()

        # save statistics
        losses[i] = eff_loss.detach().numpy()
        exp_returns[i] = np.mean(return_fht)
        var_returns[i] = np.var(return_fht)
        exp_time_steps[i] = avg_len
        policy_l2_errors[i] = policy_l2_error
        cts[i] = ct

        msg = 'it.: {:2d}, loss: {:.3e}, exp return: {:.3e}, var return: {:.1e}, ' \
              'avg ts: {:.3e}, policy l2-error: {:.2e}, ct: {:.3f}' \
              ''.format(
                  i,
                  losses[i],
                  exp_returns[i],
                  var_returns[i],
                  avg_len,
                  policy_l2_error,
                  ct,
              )
        print(msg)

        # update figure
        if plot:
            pass
            #update_det_policy_figure(env, controls[i], control_line)

        # save model
        if backup_freq_iterations is not None and (i+1) % backup_freq_iterations == 0:
            save_model(model, rel_dir_path, 'model_n-it{}'.format(i+1))

    data = {
        'rel_dir_path': rel_dir_path,
        'n_iterations': n_iterations,
        'backup_freq_iterations': backup_freq_iterations,
        'batch_size': batch_size,
        'losses': losses,
        'exp_returns': exp_returns,
        'var_returns': var_returns,
        'exp_time_steps': exp_time_steps,
        'policy_l2_errors': policy_l2_errors,
        'cts': cts,
        'model': model,
    }
    save_data(data, rel_dir_path)
    return data

def get_policy(env, model):
    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    with torch.no_grad():
        policy = model.forward(state_space_h).numpy().squeeze()
    return policy

def get_policies(env, data):

    rel_dir_path = data['rel_dir_path']
    model = data['model']
    n_iterations = data['n_iterations']
    backup_freq_iterations = data['backup_freq_iterations']

    Nx = env.n_states
    policies = np.empty((0, Nx), dtype=np.float32)

    for i in range(data['n_iterations']):
        if i == 0:
            load_model(model, rel_dir_path, file_name='model_n-it{}'.format(0))
            policies = np.vstack((policies, get_policy(env, model).reshape(1, Nx)))

        elif (i + 1) % backup_freq_iterations == 0:
            load_model(model, rel_dir_path, file_name='model_n-it{}'.format(i+1))
            policies = np.vstack((policies, get_policy(env, model).reshape(1, Nx)))

    return policies



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

    # run reinforve algorithm with a deterministic policy
    data = reinforce(
        env,
        gamma=args.gamma,
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

    # plot expected values for each epoch
    plot_expected_returns_with_error_epochs(data['exp_returns'], data['var_returns'])
    plot_time_steps_epochs(data['avg_time_steps'])

    # plot policy l2 error
    plot_det_policy_l2_error_epochs(data['policy_l2_errors'])

    # plot loss function
    plot_loss_epochs(data['losses'])

    # plot policy
    policies = get_policies(env, data)
    plot_det_policies(env, policies, sol_hjb.u_opt)
    #plot_det_policies_black_and_white(env, policies, sol_hjb.u_opt)
    policy = get_policy(env, data['model'])
    plot_det_policy(env, policy, sol_hjb.u_opt)

    return


    losses = data['losses']
    plot_losses(losses)

if __name__ == "__main__":
    main()
