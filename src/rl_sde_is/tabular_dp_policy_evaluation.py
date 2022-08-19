import numpy as np

from sde.langevin_sde import LangevinSDE
from hjb.hjb_solver import SolverHJB

from base_parser import get_base_parser
from dynammic_programming import compute_p_tensor_batch, compute_r_table, \
                                 plot_policy, plot_value_function
from environments import DoubleWellStoppingTime1D

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def check_bellman_equation():
    pass
    # check that p_tensor and r_table are well computed
    #a = -hjb_value_f[idx_x_init]
    #b = np.dot(
    #   p_tensor[np.arange(env.n_states), idx_x_init, policy[idx_x_init]],
    #   - hjb_value_f[np.arange(env.n_states)],
    #) + rew
    #assert a == b, ''

def policy_evaluation(env, policy, hjb_value_f, gamma=1.0,
                      n_iterations=100, n_avg_iterations=10):
    '''
    '''
    # compute p tensor and r table
    p_tensor = compute_p_tensor_batch(env)
    r_table = compute_r_table(env)

    # initialize value function table
    v_table = np.random.rand(env.n_states)

    # set values for the target set
    v_table[env.idx_lb:env.idx_rb+1] = 0

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init)

    # for each iteration
    for i in np.arange(n_iterations):

        # copy value function table
        v_table_i = v_table.copy()

        for idx_state in range(env.idx_lb):
            idx_action = policy[idx_state]

            value = r_table[idx_state, idx_action]

            for idx_next_state in range(env.n_states):
                value += gamma * p_tensor[idx_next_state, idx_state, idx_action] \
                               * v_table[idx_next_state]

            v_table[idx_state] = value


        # logs
        if i % n_avg_iterations == 0:
            msg = 'it: {:3d}, V(s_init): {:.3f}'.format(i, v_table[idx_state_init])
            print(msg)

    return v_table

def main():
    args = get_parser().parse_args()

    # initialize environments
    env = DoubleWellStoppingTime1D()

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get target set indices
    env.get_idx_target_set()

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
        env.get_action_idx(sol_hjb.u_opt[::k][idx_state])
        for idx_state, _ in enumerate(env.state_space_h)
    ])

    # run mc learning agent following optimal policy
    v_table = policy_evaluation(
        env,
        policy,
        sol_hjb.value_function[::k],
        gamma=args.gamma,
        n_iterations=args.n_iterations,
        n_avg_iterations=args.n_avg_iterations,
    )

    # do plots
    plot_policy(env, policy, control_hjb=sol_hjb.u_opt[::k])
    plot_value_function(env, v_table, value_f_hjb=sol_hjb.value_function[::k])


if __name__ == '__main__':
    main()
