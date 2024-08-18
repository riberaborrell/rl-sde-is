import gymnasium as gym

import gym_sde_is

from rl_sde_is.discrete.dqn_core import dqn
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.tabular_methods import *#compute_value_advantage_and_greedy_policy
from rl_sde_is.utils.approximate_methods import evaluate_qvalue_function_discrete_model
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.description = 'Run dqn for the sde importance sampling environment \
                          with a ol toy example.'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        state_init_dist=args.state_init_dist,
    )

    # discretize action space
    h_coarse = 0.1
    env.discretize_state_space(h_coarse)
    env.discretize_action_space(args.h_action)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    sol_hjb.coarse_solution(h_coarse)

    # run dqn 
    data = dqn(
        env=env,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        n_episodes=args.n_episodes,
        replay_size=args.replay_size,
        learning_starts=args.learning_starts,
        n_steps_lim=args.n_steps_lim,
        update_freq=args.update_freq,
        polyak=args.polyak,
        seed=args.seed,
        load=args.load
    )
    n_episodes = data['n_episodes']
    returns = data['returns']
    time_steps = data['time_steps']
    loss = data['losses']
    breakpoint()

    # plot returns and time steps
    x = np.arange(n_episodes)
    plot_y_per_episode(x, returns, run_window=10, title='Returns', legend=True)
    plot_y_per_episode(x, time_steps, run_window=10, title='Time steps')

    # plot value, q-value and advantage functions
    q_table = evaluate_qvalue_function_discrete_model(env, data['model'])
    v_table = compute_value_function(q_table)
    a_table = compute_advantage_function(v_table, q_table)
    actions_idx = compute_greed_action_indices(q_table)
    greedy_policy = env.action_space_h[actions_idx]
    plot_value_function_1d(env, v_table, -sol_hjb.value_function)
    plot_q_value_function_1d(env, q_table)
    plot_advantage_function_1d(env, a_table)
    plot_det_policy_1d(env, greedy_policy, sol_hjb.u_opt)

if __name__ == '__main__':
    main()
