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

def td_prediction(env, policy=None, gamma=1.0, n_episodes=100, lr=0.01,
                  n_steps_lim=1000, test_freq_episodes=10, seed=None,
                  value_function_opt=None, load=False, live_plot=False):

    ''' Temporal difference learning for policy evaluation.
    '''

    # get dir path
    rel_dir_path = get_tabular_td_prediction_dir_path(
        env,
        agent='tabular-td-prediction',
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

    # initialize value function table
    v_table = - np.random.rand(env.n_states)

    # set values for the target set
    v_table[env.idx_ts] = 0

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init).item()

    # preallocate value function rms errors
    n_test_episodes = n_episodes // test_freq_episodes + 1
    v_rms_errors = np.empty(n_test_episodes)

    # initialize live figures
    if live_plot:
        line = initialize_value_function_1d_figure(env, v_table, value_function_opt)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # get index of the state
        idx_state = env.get_state_idx(state)

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
            idx_action = policy[idx_state]
            action = env.action_space_h[idx_action]

            # step dynamics forward
            new_state, r, done, _ = env.step(state, action)
            idx_new_state = env.get_state_idx(new_state)

            # update v values
            v_table[idx_state] += lr * (
                r + gamma * v_table[idx_new_state] - v_table[idx_state]
            )

            # save reward
            rewards = np.append(rewards, r)

            # update state and action
            state = new_state
            idx_state = idx_new_state

        # test
        if (ep + 1) % test_freq_episodes == 0:

            # compute root mean square error of value function
            ep_test = (ep + 1) // test_freq_episodes
            v_rms_errors[ep_test] = compute_rms_error(value_function_opt, v_table)

            # logs
            msg = 'ep: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}'.format(
                    ep,
                    v_table[idx_state_init],
                    v_rms_errors[ep_test],
                )
            print(msg)

            # update live figure
            if live_plot:
                update_value_function_1d_figure(env, v_table, line)

    data = {
        'gamma': gamma,
        'n_episodes': n_episodes,
        'n_steps_lim': n_steps_lim,
        'lr': lr,
        'seed': seed,
        'test_freq_episodes' : test_freq_episodes,
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
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # set deterministic policy from the hjb control
    policy = np.array([
        env.get_action_idx(sol_hjb.u_opt[idx_state])
        for idx_state, _ in enumerate(env.state_space_h)
    ])

    # run temporal difference learning agent following optimal policy
    data = td_prediction(
        env,
        policy=policy,
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
    policy = env.action_space_h[policy]
    plot_det_policy_1d(env, policy, sol_hjb.u_opt)
    plot_value_function_1d(env, data['v_table'], -sol_hjb.value_function)
    plot_value_rms_error_episodes(data['v_rms_errors'], data['test_freq_episodes'])

if __name__ == '__main__':
    main()
