import numpy as np

from rl_sde_is.base_parser import get_base_parser
from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.plots import *
from rl_sde_is.tabular_methods import compute_rms_error
from rl_sde_is.utils_path import *

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser


def mc_prediction(env, policy=None, gamma=1.0, n_episodes=100,
                  n_steps_lim=1000, test_freq_episodes=10, first_visit=False, seed=None,
                  value_function_opt=None, load=False, live_plot=False):

    ''' Monte Carlo learning for policy evaluation. First-visit and every-visit
        implementation (Sutton and Barto)
    '''

    # get dir path
    rel_dir_path = get_tabular_mc_prediction_dir_path(
        env,
        agent='tabular-mc-prediction',
        n_episodes=n_episodes,
        seed=seed,
    )

    # load results
    if load:
        return load_data(rel_dir_path)

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # initialize value function table and returns
    v_table = -np.random.rand(env.n_states)
    returns_table = [[] for i in range(env.n_states)]

    # set values for the target set
    v_table[env.ts_idx] = 0

    # preallocate returns and time steps
    time_steps = np.empty(n_episodes, dtype=np.int32)

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

        # reset trajectory
        states = np.empty(0)
        rewards = np.empty(0)

        states_idx = []

        # terminal state flag
        done = False

        # sample episode
        for k in np.arange(n_steps_lim):

            # interrupt if we are in a terminal state
            if done:
                break

            # save state
            states = np.append(states, state)

            # get index of the state
            state_idx = env.get_state_idx(state)
            states_idx.append(state_idx.item())

            # choose action following the given policy
            action_idx = policy[state_idx]
            action = np.expand_dims(env.action_space_h[action_idx], axis=1)

            # step dynamics forward
            new_state, r, done, _ = env.step(state, action)

            # save reward
            rewards = np.append(rewards, r)

            # update state
            state = new_state

        # save number of time steps of the episode
        time_steps[ep] = k

        ret = 0
        for k in np.flip(np.arange(time_steps[ep])):

            # compute return at time step k  
            ret = gamma * ret + rewards[k]

            # if state not in the previous time steps
            if not states_idx[k] in states_idx[:k] or not first_visit:

                # get current state index
                state_idx = states_idx[k]

                # append return
                returns_table[state_idx].append(ret)

                # update v table
                v_table[state_idx] = np.mean(returns_table[state_idx])

        # test
        if (ep + 1) % test_freq_episodes == 0:

            # compute root mean square error of value function
            ep_test = (ep + 1) // test_freq_episodes
            v_rms_errors[ep_test] = compute_rms_error(value_function_opt, v_table)

            # logs
            msg = 'ep: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}'.format(
                ep,
                v_table[env.state_init_idx.item()],
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

    # set action space bounds
    env.set_action_space_bounds()

    # discretize observation and action space
    env.discretize_state_space(args.h_state)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()

    # set deterministic policy from the hjb control
    policy = env.get_det_policy_indices_from_hjb(sol_hjb.u_opt)

    # run mc value function learning agent following optimal policy
    data = mc_prediction(
        env,
        policy=policy,
        gamma=args.gamma,
        n_steps_lim=args.n_steps_lim,
        n_episodes=args.n_episodes,
        test_freq_episodes=args.test_freq_episodes,
        first_visit=False,
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
