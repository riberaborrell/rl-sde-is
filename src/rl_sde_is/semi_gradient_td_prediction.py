import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.approximate_methods import *
from rl_sde_is.tabular_methods import *
from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import mlp
from rl_sde_is.plots import *
from rl_sde_is.utils_path import *

class ValueFunction(nn.Module):

    def __init__(self, state_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim] + list(hidden_sizes) + [1]
        self.v = mlp(self.sizes, activation)
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-4, 5e-4)
                nn.init.uniform_(module.bias, -5e-4, 5e-4)

    def forward(self, state):
        return self.v(state)

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def td_prediction(env, policy=None, gamma=1.0, n_episodes=100, lr=0.01,
                  n_steps_lim=1000, test_freq_episodes=10, seed=None,
                  value_function_opt=None, load=False, live_plot=False):

    ''' Temporal difference learning for policy evaluation.
    '''

    # get dir path
    rel_dir_path = get_semi_gradient_td_prediction_dir_path(
        env,
        agent='semi-gradient-td-prediction',
        n_episodes=n_episodes,
        lr=lr,
        seed=seed,
    )

    # load results
    if load:
        data = load_data(rel_dir_path)
        return data

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # initialize value function model
    model = ValueFunction(
        state_dim=env.state_space_dim,
        hidden_sizes=[32, 32],
        activation=nn.Tanh()
    )

    # set optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init).item()

    # preallocate value function rms errors
    n_test_episodes = n_episodes // test_freq_episodes + 1
    v_rms_errors = np.empty(n_test_episodes)

    # initialize live figures
    if live_plot:
        value_function = compute_v_table(env, model)
        line = initialize_value_function_1d_figure(env, value_function, value_function_opt)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = torch.FloatTensor(env.reset(batch_size=1))

        # reset trajectory rewards
        rewards = np.empty(0)

        # terminal state flag
        done = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # choose action following the given policy
            idx_state = env.get_state_idx(state)
            action = torch.FloatTensor(policy[[idx_state]])

            # step dynamics forward
            new_state, r, done, _ = env.step_torch(state, action)

            # reset gradients
            optimizer.zero_grad()

            # loss
            MSE = nn.MSELoss()
            state_value = model.forward(state)
            with torch.no_grad():
                prediction = model.forward(new_state)
                target = r + gamma * prediction
            loss = MSE(state_value, target)
            loss.backward()

            # update parameters
            optimizer.step()

            # save reward
            rewards = np.append(rewards, r)

            # update state and action
            state = new_state

        # test
        if (ep + 1) % test_freq_episodes == 0:

            # compute root mean square error of value function
            ep_test = (ep + 1) // test_freq_episodes
            value_function = compute_v_table(env, model).squeeze()
            v_rms_errors[ep_test] = compute_rms_error(value_function, value_function_opt)

            # logs
            msg = 'ep: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}'.format(
                   ep,
                   value_function[idx_state_init],
                   v_rms_errors[ep_test],
                )
            print(msg)

            # update live figures
            if live_plot:
                update_value_function_1d_figure(env, value_function, line)

    data = {
        'n_episodes': n_episodes,
        'lr': lr,
        'seed': seed,
        'v_rms_errors' : v_rms_errors,
        'model': model,
    }
    save_data(data, rel_dir_path)

    return data

def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize state space
    env.discretize_state_space(h_state=0.01)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run semi-gradient temporal difference learning agent following optimal policy
    data = td_prediction(
        env,
        policy=sol_hjb.u_opt,
        gamma=args.gamma,
        lr=args.lr,
        n_steps_lim=args.n_steps_lim,
        n_episodes=args.n_episodes,
        test_freq_episodes=args.test_freq_episodes,
        seed=args.seed,
        value_function_opt=-sol_hjb.value_function,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plot
    if not args.plot:
        return

    # do plots
    value_function = compute_v_table(env, data['model'])
    plot_value_function_1d(env, value_function, -sol_hjb.value_function)
    plot_value_rms_error_episodes(data['v_rms_errors'], args.test_freq_episodes)

if __name__ == '__main__':
    main()
