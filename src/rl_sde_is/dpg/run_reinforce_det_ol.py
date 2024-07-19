import gymnasium as gym
import gym_sde_is

from rl_sde_is.dpg.reinforce_deterministic_core import *
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *


def main():
    parser = get_base_parser()
    parser.description = 'Run model-based reinforce for deterministic policies for the sde \
                          importance sampling environment with a ol toy example.'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=(args.alpha),
        state_init_dist=args.state_init_dist,
        is_torch=True,
    )

    # discretize state and action space (plot purposes only)
    h_coarse = 0.1
    env.discretize_state_space(h_state=h_coarse)
    env.discretize_action_space(h_action=h_coarse)

    # get hjb solver
    sol_hjb = env.get_hjb_solver()
    sol_hjb.coarse_solution(h_coarse)
    policy_opt = np.expand_dims(sol_hjb.u_opt, axis=1) if env.d == 1 else sol_hjb.u_opt

    # run reinforve algorithm with a deterministic policy
    data = reinforce_deterministic(
        env,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        n_grad_iterations=args.n_grad_iterations,
        seed=args.seed,
        backup_freq=args.backup_freq,
        live_plot_freq=args.live_plot_freq,
        policy_opt=policy_opt,
        track_l2_error=args.track_l2_error,
        load=args.load,
    )

    # do plots
    if not args.plot:
        return

    # get backup iterations
    iterations = np.arange(args.n_iterations + 1)
    iterations_backup = iterations[::data['backup_freq']]

    # plot policy
    if env.d <= 2:
        policies = get_policies(env, data, iterations_backup[::20])

    if env.d == 1:
        plot_det_policies_1d(env, policies, sol_hjb.u_opt)

    if env.d == 2:
        plot_det_policy_2d(env, policies[0].reshape(env.n_states_axis+(env.d,)), policy_opt)
        plot_det_policy_2d(env, policies[-1].reshape(env.n_states_axis+(env.d,)), policy_opt)

if __name__ == "__main__":
    main()
