import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import *
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def td_prediction(env, policy=None, batch_size=1000, gamma=1.0, lr=0.01,
                  n_steps_lim=int(1e6), n_total_steps=int(1e6), test_freq_steps=int(1e3), seed=None,
                  value_function_opt=None, load=False, live_plot=False):

    ''' Temporal difference learning for policy evaluation.
    '''

    # get dir path
    rel_dir_path = get_tabular_td_prediction_dir_path(
        env,
        agent='tabular-td-prediction-batch',
        n_episodes=n_total_steps,
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

    # initialize value function table
    v_table = - np.random.rand(env.n_states)

    # get index initial state
    state_init_idx = env.get_state_idx(env.state_init).item()

    # preallocate value function rms errors
    n_test_steps = n_total_steps // test_freq_steps + 1
    v_rms_errors = np.empty(n_test_steps)

    # initialize live figures
    if live_plot:
        line = initialize_value_function_1d_figure(env, v_table, value_function_opt)

    # reset environment
    state = env.reset(batch_size)

    # terminal state flag
    done = np.full((batch_size, 1), False)

    # sample episode
    for k in np.arange(n_total_steps):

        # reset if we are in a terminal state
        env.reset_done(state, done)

        # choose action following the given policy
        state_idx = env.get_state_idx(state)
        action_idx = policy[state_idx]
        action = np.expand_dims(env.action_space_h[action_idx], axis=1)

        # step dynamics forward
        next_state, r, done, _ = env.step(state, action)
        next_state_idx = env.get_state_idx(next_state)

        # update v values
        d = np.where(done, 1., 0.)
        target = r + gamma * (1 - d) * v_table[next_state_idx]
        v_table[state_idx] += lr * (target - v_table[state_idx])

        # update state and action
        state = next_state

        # test
        if (k + 1) % test_freq_steps == 0:

            # compute root mean square error of value function
            k_test = (k + 1) // test_freq_steps
            v_rms_errors[k_test] = compute_rms_error(value_function_opt, v_table)

            # logs
            msg = 'k: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}'.format(
                k+1,
                v_table[state_init_idx],
                v_rms_errors[k_test],
            )
            print(msg)

            # update live figure
            if live_plot:
                update_value_function_1d_figure(env, v_table, line)

    data = {
        'gamma': gamma,
        'batch_size': batch_size,
        'n_steps_lim': n_steps_lim,
        'n_total_steps': n_total_steps,
        'lr': lr,
        'seed': seed,
        'test_freq_steps' : test_freq_steps,
        'v_table' : v_table,
        'v_rms_errors' : v_rms_errors,
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

    # discretize observation and action space
    env.set_action_space_bounds()
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # set deterministic policy from the hjb control
    policy = env.get_det_policy_indices_from_hjb(sol_hjb.u_opt)

    # run temporal difference learning agent following optimal policy
    data = td_prediction(
        env,
        policy=policy,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        n_steps_lim=args.n_steps_lim,
        n_total_steps=args.n_total_steps,
        #test_freq_steps=args.test_freq_steps,
        seed=args.seed,
        value_function_opt=-sol_hjb.value_function,
        load=args.load,
        live_plot=args.live_plot,
    )

    # plot
    if not args.plot:
        return

    # do plots
    policy = env.action_space_h[policy]
    plot_det_policy_1d(env, policy, sol_hjb.u_opt)
    plot_value_function_1d(env, data['v_table'], -sol_hjb.value_function)
    plot_value_rms_error_episodes(data['v_rms_errors'], data['test_freq_episodes'])

if __name__ == '__main__':
    main()
