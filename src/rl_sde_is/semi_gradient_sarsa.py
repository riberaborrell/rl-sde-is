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

class QValueFunction(nn.Module):

    def __init__(self, state_dim, n_actions, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim] + list(hidden_sizes) + [n_actions]
        self.q = mlp(self.sizes, activation)
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-4, 5e-4)
                nn.init.uniform_(module.bias, -5e-4, 5e-4)

    def forward(self, state):
        return self.q(state)

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def get_epsilon_greedy_action(env, model, epsilon, state):

    # pick greedy action (exploitation)
    if np.random.rand() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state)
            phi = model.forward(state_tensor)
            #return np.expand_dims(torch.max(phi).numpy(), axis=0)
            return torch.unsqueeze(torch.max(phi), dim=0)

    # pick random action (exploration)
    else:
        #return torch.random.rand(env.action_space_bounds[0], env.action_space_bounds[1], (1,))
        a = env.action_space_bounds[0]
        b = env.action_space_bounds[1]
        return (a - b) * torch.rand((1,)) + b

def sarsa(env, gamma=1.0, epsilons=None, lr=0.01,
          n_episodes=100, n_avg_episodes=10, n_steps_lim=1000, seed=None,
          policy=None, value_function_opt=None, load=False, live_plot=False):

    '''
    '''

    # get dir path
    #rel_dir_path = get_tabular_td_prediction_dir_path(
    #    env,
    #    agent='semi-gradient-td-prediction',
    #    n_episodes=n_episodes,
    #    lr=lr,
    #)

    # load results
    #if load:
    #    data = load_data(rel_dir_path)
    #    return data

    # initialize q value function model
    model = QValueFunction(
        state_dim=env.state_space_dim,
        n_actions=env.n_actions,
        hidden_sizes=[32, 32],
        activation=nn.Tanh()
    )

    # set optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init).item()

    # preallocate returns and time steps
    returns = np.empty(n_episodes)
    avg_returns = np.empty(n_episodes)
    time_steps = np.empty(n_episodes, dtype=np.int32)
    avg_time_steps = np.empty(n_episodes)

    # preallocate value function and control rms errors
    v_rms_errors = np.empty(n_episodes)
    p_rms_errors = np.empty(n_episodes)

    # initialize live figures
    if live_plot:
        value_function = compute_v_table(env, model)
        line = initialize_value_function_1d_figure(env, value_function, value_function_opt)

    # for each episode
    for ep in np.arange(n_episodes):
        print(ep)

        # get epsilon
        epsilon = epsilons[ep]

        # reset environment
        state = torch.FloatTensor(env.reset(batch_size=1))
        action = get_epsilon_greedy_action(env, model, epsilon, state)
        #action = np.expand_dims(action, axis=0)
        action = torch.unsqueeze(action, dim=1)
        idx_action = env.get_action_idx(action)

        # reset trajectory rewards
        rewards = np.empty(0)

        # terminal state flag
        complete = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                break

            # step dynamics forward
            #new_state, r, complete, _ = env.step_torch(state, action)
            new_state, r, complete, _ = env.step_torch(state, action)

            # get new action
            new_action = get_epsilon_greedy_action(env, model, epsilon, new_state)
            #new_action = np.expand_dims(new_action, axis=0)
            new_action = torch.unsqueeze(new_action, dim=1)
            idx_new_action = env.get_action_idx(new_action)

            # choose action following the given policy
            #idx_state = env.get_state_idx(state)
            #action = torch.FloatTensor(policy[[idx_state]])

            # reset gradients
            optimizer.zero_grad()

            # loss
            MSE = nn.MSELoss()
            q_value = model.forward(state[0])[idx_action]
            with torch.no_grad():
                prediction = model.forward(new_state[0])[idx_new_action]
                target = r + gamma * prediction
            loss = MSE(q_value, target)
            loss.backward()

            print(loss.item())
            # update parameters
            optimizer.step()

            # save reward
            rewards = np.append(rewards, r)

            # update state and action
            state = new_state
            action = new_action
            idx_action = idx_new_action

        # compute returns at each time step
        ep_returns = discount_cumsum(rewards, gamma)

        # get indices episodes to averaged
        if ep < n_avg_episodes:
            idx_last_episodes = slice(0, ep + 1)
        else:
            idx_last_episodes = slice(ep + 1 - n_avg_episodes, ep + 1)

        # save episode
        returns[ep] = ep_returns[0]
        avg_returns[ep] = np.mean(returns[idx_last_episodes])
        time_steps[ep] = rewards.shape[0]
        avg_time_steps[ep] = np.mean(time_steps[idx_last_episodes])

        # compute root mean square error of value function
        #value_function = compute_v_table(env, model).squeeze()
        #v_rms_errors[ep] = compute_rms_error(value_function, value_function_opt)

        # logs
        #if ep % n_avg_episodes == 0:
        #    msg = 'ep: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}'.format(
        #           ep,
        #           value_function[idx_state_init],
        #           v_rms_errors[ep],
        #        )
        #    print(msg)

        # update live figures
        if live_plot and ep % n_avg_episodes == 0:
            update_value_function_1d_figure(env, value_function, line)

    data = {
        'n_episodes': n_episodes,
        #'v_rms_errors' : v_rms_errors,
    }
    #save_data(data, rel_dir_path)

    #return data
    return data, model

def main():
    args = get_parser().parse_args()

    # initialize environment
    env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)

    # set explorable starts flag
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretize action space
    env.discretize_action_space(args.h_action)

    # discretize state space
    env.discretize_state_space(h_state=0.01)

    # set epsilons
    #epsilons = get_epsilons_constant(args.n_episodes, eps_init=0.)
    #epsilons = get_epsilons_constant(args.n_episodes, eps_init=1.)
    epsilons = get_epsilons_linear_decay(args.n_episodes, eps_min=0.01, exploration=0.5)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # run semi-gradient temporal difference learning agent following optimal policy
    #data = sarsa(
    data, model = sarsa(
        env,
        gamma=args.gamma,
        epsilons=epsilons,
        lr=args.lr,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        n_steps_lim=args.n_steps_lim,
        seed=args.seed,
        value_function_opt=-sol_hjb.value_function,
        policy=sol_hjb.u_opt,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plot
    if not args.plot:
        return

    # do plots
    #q_table = compute_q_table(env, model)
    #v_table, a_table, policy_greedy = compute_tables(env, q_table)
    q_table, v_table, a_table, policy_greedy = compute_tables_discrete_actions(env, model)
    plot_value_function_1d(env, v_table, -sol_hjb.value_function)
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table)
    plot_det_policy_1d(env, policy_greedy, sol_hjb.u_opt)
    #plot_value_rms_error_episodes(data['v_rms_errors'])
    #plot_policy_rms_error_epochs(data['p_rms_errors'])

if __name__ == '__main__':
    main()
