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


def mc_prediction(env, gamma=1.0, n_episodes=100, n_avg_episodes=10, n_steps_lim=1000,
                  first_visit=False, seed=None, policy=None, value_function_opt=None, load=False):

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
        data = load_data(rel_dir_path)
        return data

    # set seed
    if seed is not None:
        np.random.seed(seed)

    # initialize value function table and returns
    v_table = -np.random.rand(env.n_states)
    returns_table = [[] for i in range(env.n_states)]

    # set values for the target set
    v_table[env.idx_lb:env.idx_rb+1] = 0

    # get index initial state
    idx_state_init = env.get_state_idx(env.state_init).item()

    # preallocate returns and time steps
    time_steps = np.empty(n_episodes, dtype=np.int32)

    # preallocate value function rms errors
    v_rms_errors = np.empty(n_episodes)

    # for each episode
    for ep in np.arange(n_episodes):

        # reset environment
        state = env.reset()

        # reset trajectory
        states = np.empty(0)
        rewards = np.empty(0)

        idx_states = []

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
            idx_state = env.get_state_idx(state)
            idx_states.append(idx_state.item())

            # choose action following the given policy
            idx_action = policy[idx_state]
            action = env.action_space_h[idx_action]

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
            ret = gamma * ret + r

            # if state not in the previous time steps
            if not idx_states[k] in idx_states[:k] or not first_visit:

                # get current state index
                idx_state = idx_states[k]

                # append return
                returns_table[idx_state].append(ret)

                # update v table
                v_table[idx_state] = np.mean(returns_table[idx_state])

        # compute root mean square error of value function
        v_rms_errors[ep] = compute_rms_error(value_function_opt, v_table)

        # logs
        if ep % n_avg_episodes == 0:
            msg = 'ep: {:3d}, V(s_init): {:.3f}, V_RMSE: {:.3f}'.format(
                    ep,
                    v_table[idx_state_init],
                    v_rms_errors[ep],
                )
            print(msg)

    data = {
        'n_episodes': n_episodes,
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

    # run mc value function learning agent following optimal policy
    data = mc_prediction(
        env,
        policy=policy,
        value_function_opt=-sol_hjb.value_function,
        gamma=args.gamma,
        n_steps_lim=args.n_steps_lim,
        n_episodes=args.n_episodes,
        n_avg_episodes=args.n_avg_episodes,
        first_visit=True,
        seed=args.seed,
        load=args.load,
    )

    # do plots
    policy = env.action_space_h[policy]
    #plot_det_policy_1d(env, policy, sol_hjb.u_opt)
    plot_value_function_1d(env, data['v_table'], sol_hjb.value_function)
    plot_value_rms_error_epochs(data['v_rms_errors'])



if __name__ == '__main__':
    main()
