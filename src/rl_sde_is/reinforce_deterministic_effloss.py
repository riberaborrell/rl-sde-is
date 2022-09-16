import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from base_parser import get_base_parser
from environments import DoubleWellStoppingTime1D
from models import FeedForwardNN, DenseNN
from plots import *
from utils_path import *

def get_parser():
    parser = get_base_parser()
    return parser

def sample_loss_vectorized(env, model, K):

    # max time steps
    dt = env.dt_tensor
    k_max = 10**6

    # start timer
    ct_initial = time.time()

    # initialize trajectories
    states = torch.FloatTensor(env.reset_vectorized(batch_size=K))

    # initialize work and running integral
    work_t = torch.zeros(K)
    work_fht = torch.empty(K)
    det_int_t = torch.zeros(K)
    det_int_fht = torch.empty(K)
    stoch_int_t = torch.zeros(K)
    stoch_int_fht = torch.empty(K)

    # preallocate time steps
    time_steps = np.empty(K)

    been_in_target_set = torch.full((K, 1), False)
    done = torch.full((K, 1), False)

    for k in np.arange(1, k_max + 1):

        # actions
        actions = model.forward(states)

        # step dynamics forward
        next_states, rewards, done, dbt = env.step_vectorized_torch(states, actions)

        # update work with running cost
        work_t = work_t + env.f(states) * dt

        # update deterministic integral
        det_int_t = det_int_t + (torch.linalg.norm(actions, axis=1) ** 2) * dt

        # update stochastic integral
        stoch_int_t = stoch_int_t + torch.matmul(
            actions[:, np.newaxis, :],
            dbt[:, :, np.newaxis],
        ).squeeze()

        # get indices of trajectories which are new to the target set
        idx = env.get_idx_new_in_ts_torch(done, been_in_target_set)

        if idx.shape[0] != 0:

            # update work with final cost
            work_t = work_t + env.g(states)

            # fix work running integral
            work_fht[idx] = work_t.index_select(0, idx)

            # fix running integrals
            det_int_fht[idx] = det_int_t.index_select(0, idx)
            stoch_int_fht[idx] = stoch_int_t.index_select(0, idx)

            # time steps
            time_steps[idx] = k

        # stop if xt_traj in target set
        if been_in_target_set.all() == True:
           break

        # update states
        states = next_states

    # compute cost functional (loss)
    phi_fht = (work_fht + 0.5 * det_int_fht).detach()

    # compute effective loss
    eff_loss = torch.mean(0.5 * det_int_fht + phi_fht * stoch_int_fht)
    #eff_loss = torch.mean(phi_fht * stoch_int_fht)

    # compute mean and re of I_u
    I_u = np.exp(
        - work_fht.numpy()
        - stoch_int_fht.detach().numpy()
        - 0.5 * det_int_fht.detach().numpy()
    )
    mean_I_u = np.mean(I_u)
    var_I_u = np.var(I_u)
    re_I_u = np.sqrt(var_I_u) / mean_I_u

    # end timer
    ct_final = time.time()

    return eff_loss, phi_fht.numpy(), mean_I_u, var_I_u, re_I_u, time_steps, ct_final - ct_initial

def reinforce(env, gamma=1.0, n_layers=3, d_hidden_layer=30, is_dense=False,
              batch_size=10, lr=0.01, n_iterations=100, seed=None,
              control_hjb=None, load=False, plot=False):

    # get dir path
    dir_path = get_reinforce_det_dir_path(
        env,
        agent='reinforce-det',
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

    # initialize nn model 
    model = FeedForwardNN(d_in=env.state_space_dim, hidden_sizes=d_hidden_layers,
                          d_out=env.action_space_dim, activation=nn.Tanh())

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # number of discretized states
    Nh = env.state_space_h.shape[0]

    # preallocate parameters
    losses = np.empty(n_iterations)
    var_losses = np.empty(n_iterations)
    controls = np.empty((n_iterations, Nh))
    means_I_u = np.empty(n_iterations)
    vars_I_u = np.empty(n_iterations)
    res_I_u = np.empty(n_iterations)
    #avg_time_steps = np.empty(n_iterations, dtype=np.int64)
    avgs_time_steps = np.empty(n_iterations)
    cts = np.empty(n_iterations)

    # save initial control
    state_space_h = torch.FloatTensor(env.state_space_h).unsqueeze(dim=1)
    controls[0] = model.forward(state_space_h).detach().numpy().squeeze()

    # initialize animated figures
    if plot:
        control_line = initialize_det_policy_figure(env, controls[0], control_hjb)

    for i in np.arange(n_iterations):

        # reset gradients
        optimizer.zero_grad()

        # compute effective loss and relative entropy loss (phi_fht)
        eff_loss, phi_fht, mean_I_u, var_I_u, re_I_u, time_steps, ct = sample_loss_vectorized(env, model, batch_size)
        eff_loss.backward()

        # update parameters
        optimizer.step()

        # compute loss and variance
        loss = np.mean(phi_fht)
        var = np.var(phi_fht)

        # average time steps
        avg_time_steps = np.mean(time_steps)

        msg = 'it.: {:2d}, loss: {:.3e}, var: {:.1e}, mean I^u: {:.3e}, var I^u: {:.1e}, ' \
              're I^u: {:.1e}, avg ts: {:.3e}, ct: {:.3f}' \
              ''.format(i, loss, var, mean_I_u, var_I_u, re_I_u, avg_time_steps, ct)
        print(msg)

        # save statistics
        losses[i] = loss
        var_losses[i] = var
        means_I_u[i] = mean_I_u
        vars_I_u[i] = var_I_u
        res_I_u[i] = re_I_u
        avgs_time_steps[i] = avg_time_steps
        cts[i] = ct

        # save control
        controls[i] = model.forward(state_space_h).detach().numpy().squeeze()

        # update figure
        if plot:
            update_det_policy_figure(env, controls[i], control_line)

    data = {
        'n_iterations': n_iterations,
        'batch_size': batch_size,
        'controls': controls,
        'losses': losses,
        'var_losses': var_losses,
        'means_I_u': means_I_u,
        'vars_I_u': vars_I_u,
        'res_I_u': res_I_u,
        'avg_time_steps': avg_time_steps,
        'cts': cts,
    }
    save_data(dir_path, data)
    return data

def plot_controls(env, controls, u_hjb):
    n_controls = controls.shape[0]

    fig, ax = plt.subplots()
    for i in range(n_controls):
        if i % 10 == 0:
            ax.plot(env.state_space_h, controls[i])
        ax.plot(env.state_space_h, u_hjb, color='cyan')
    ax.set_ylim(-3, 3)
    plt.show()

def plot_losses(losses):
    fig, ax = plt.subplots()
    ax.plot(losses)
    plt.show()

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
        seed=args.seed,
        control_hjb=sol_hjb.u_opt,
        load=args.load,
        plot=args.plot,
    )
    losses = data['losses']
    controls = data['controls']

    # do plots
    if args.plot:
        plot_losses(losses)
        #plot_controls(env, controls, sol_hjb.u_opt)
        plot_det_policy(env, controls[-1], sol_hjb.u_opt)

if __name__ == "__main__":
    main()
