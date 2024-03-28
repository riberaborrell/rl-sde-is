import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.models import mlp, GaussianAnsatzModel
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.environments_2d import DoubleWellStoppingTime2D
from rl_sde_is.reinforce_deterministic_core import *
from rl_sde_is.plots import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def reinforce(env, gamma=1.00, d_hidden_layer=32, n_layers=3,
              lr=0.01, n_episodes=2000, seed=None,
              policy_opt=None, live_plot=None):

    # set seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # get dimensions of each layer
    d_hidden_layers = [d_hidden_layer for i in range(n_layers-1)]

    # initialize policy
    model = DeterministicPolicy(
        state_dim=env.state_space_dim,
        action_dim=env.action_space_dim,
        hidden_sizes=d_hidden_layers,
        activation=nn.Tanh(),
    )
    #model = GaussianAnsatzModel(env, m_i=20, sigma_i=0.5, normalized=True, seed=seed)

    # preallocate lists to hold results
    total_returns = []
    total_time_steps = []

    # define optimizer
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # initialize live figures
    if live_plot and env.d == 1:
        policy_line = initialize_1d_figures(env, model, policy_opt)
    elif live_plot and env.d == 2:
        Q_policy = initialize_2d_figures(env, model, policy_opt)

    for ep in np.arange(n_episodes):

        # reset state
        state = torch.FloatTensor(env.reset())

        # preallocate rewards for the episode
        ep_rewards = torch.empty(0, dtype=torch.float32)

        # initialize running integrals
        stoch_int = 0.

        # time step
        n = 0

        done = False
        while done == False:

            # get action following policy
            action = model.forward(state)

            # next step
            next_state, r, done, dbt = env.step_torch(state, action)
            n += 1

            # compute running integrals
            stoch_int += torch.dot(action.squeeze(axis=0), dbt.squeeze(axis=0))

            # save rewards in trajectory
            ep_rewards = torch.cat((ep_rewards, r))

            # update states
            state = next_state

        # save statistics
        total_returns.append(sum(ep_rewards.detach().numpy()))
        total_time_steps.append(n)

        # calculate loss
        ep_return = ep_rewards.sum()
        loss = - ep_return - ep_return.detach() * stoch_int

        # reset gradients ..
        optimizer.zero_grad()

        # calculate gradients
        loss.backward()

        # update coefficients
        optimizer.step()

        # logging
        if (ep + 1) % 1 == 0:
            msg = 'ep: {}, loss: {:2.4e}, return: {:.2f}, time steps: {:.2f}'.format(
                ep + 1,
                loss,
                total_returns[-1],
                total_time_steps[-1],
            )
            print(msg)

        # update figure
        if live_plot and (ep + 1) % 10 == 0:
            if env.d == 1:
                update_1d_figures(env, model, policy_line)
            elif env.d == 2:
                update_2d_figures(env, model, Q_policy)


def main():
    args = get_parser().parse_args()

    # initialize environment
    if args.d == 1:
        env = DoubleWellStoppingTime1D(alpha=args.alpha, beta=args.beta, dt=args.dt)
    elif args.d == 2:
        env = DoubleWellStoppingTime2D(alpha=args.alpha, beta=args.beta, dt=args.dt)
    else:
        return

    # initial state sampled
    if args.explorable_starts:
        env.is_state_init_sampled = True

    # discretized state space (for plot purposes only)
    env.set_action_space_bounds()
    env.discretize_state_space(h_state=0.05)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    if args.d == 1:
        control_hjb = np.expand_dims(sol_hjb.u_opt, axis=1)
    else:
        control_hjb = sol_hjb.u_opt

    # run reinforce
    reinforce(
        env=env,
        lr=args.lr,
        n_episodes=args.n_episodes,
        seed=args.seed,
        policy_opt=control_hjb,
        live_plot=args.live_plot,
    )


if __name__ == '__main__':
    main()
