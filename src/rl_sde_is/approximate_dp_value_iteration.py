import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

from sde.langevin_sde import LangevinSDE
from hjb.hjb_solver import SolverHJB

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.models import FeedForwardNN

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def value_iteration(env, model, gamma=1.0, n_iterations=100, n_avg_iterations=10, K=10):
    '''
    '''
    low = env.state_space_low
    high = env.state_space_high

    actions = torch.FloatTensor(env.action_space_h)

    # for each iteration
    for it in np.arange(n_iterations):

        # sample training states uniformly from the domain
        states_train = torch.distributions.Uniform(low, high).sample((K, 1))

        # preallocate target values
        target_values = torch.empty_like(x)

        # get data
        for i in range(K):

            state = states_train[j]

            values = env.reward_signal(state, env.action_space_h)
            target_values[i] = r_table[idx_state, idx_action]
            #value += gamma * p_tensor[idx_next_state, idx_state, idx_action] * v_table_i[idx_next_state]



        # logs
        if it % n_avg_iterations == 0:
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(it, v_table[idx_x_init])
            print(msg)

    return v_table

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # discretize action space
    env.discretize_action_space(args.h_action)

    # initialize model
    model = FeedForwardNN(d_in=1, hidden_sizes=[32, 32], d_out=1)

    # run mc learning agent following optimal policy
    value_iteration(
        env,
        model,
        gamma=args.gamma,
        n_iterations=10,
        n_avg_iterations=1,
        K=10,
    )

    breakpoint()
    # set discretized state and action spaces
    env.state_space_h = np.arange(
        env.observation_space.low[0],
        env.observation_space.high[0] + args.h_state,
        args.h_state,
    )
    env.n_states = env.state_space_h.shape[0]

    # initialize Langevin sde
    sde = LangevinSDE(
        problem_name='langevin_stop-t',
        potential_name='nd_2well',
        d=1,
        alpha=np.ones(1),
        beta=1.,
        domain=np.full((1, 2), [-2, 2]),
    )

    # load  hjb solver
    h_hjb = 0.01
    sol_hjb = SolverHJB(sde, h=h_hjb)
    sol_hjb.load()

    # factor between the two different discretizations steps
    k = int(args.h_state / h_hjb)
    assert env.state_space_h.shape == sol_hjb.u_opt[::k, 0].shape, ''

    # set deterministic policy from the hjb control
    policy = np.array([
        get_action_idx(env, sol_hjb.u_opt[::k][idx_state])
        for idx_state, _ in enumerate(env.state_space_h)
    ])
    # do plots
    episodes = np.arange(args.n_episodes)
    plot_given_policy(env, policy, control_hjb=sol_hjb.u_opt[::k])
    plot_value_function(env, v_table, value_f_hjb=sol_hjb.value_function[::k])
    #if not agent.constant_alpha:
    #    agent.plot_alphas()




if __name__ == '__main__':
    main()
