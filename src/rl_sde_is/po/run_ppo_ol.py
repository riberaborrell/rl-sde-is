import gymnasium as gym
import gym_sde_is
import numpy as np

from rl_sde_is.po.ppo_core import PPO
from rl_sde_is.po.ppo_parser import add_ppo_arguments
from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

def main():
    parser = get_base_parser()
    parser.description = 'Run ppo for the sde importance sampling environment \
                          with a ol toy example (with hjb reference solution).'
    add_ppo_arguments(parser)
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        state_init_dist=args.state_init_dist,
    )

    # discretize state and action space (plot purposes only)
    h_coarse = 0.1
    env.unwrapped.discretize_state_space(h_state=h_coarse)
    env.unwrapped.discretize_action_space(h_action=h_coarse)

    # get hjb solver
    sol_hjb = env.unwrapped.get_hjb_solver(h_coarse)

    # PPO agent
    agent = PPO(
        env,
        gamma=args.gamma,
        policy_noise_init=args.policy_noise,
        n_layers=args.n_layers,
        #d_hidden_layer=args.d_hidden_layer,
        d_hidden_layer=args.d_hidden,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        update_epochs=args.update_epochs,
        n_mini_batches=args.n_mini_batches,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        optim_type=args.optim_type,
        norm_adv=args.norm_adv,
        clip_vloss=args.clip_vloss,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        seed=args.seed,
        cuda=args.cuda,
        torch_deterministic=args.torch_deterministic,
    )

    # run
    succ, data = agent.run_ppo(
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        live_plot_freq=args.live_plot_freq,
        policy_opt=sol_hjb.u_opt,
        value_function_opt=-sol_hjb.value_function,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # get backup iterations
    backup_iterations = np.arange(0, args.n_iterations + args.backup_freq, args.backup_freq)

    # plot statistics
    x = np.arange(data['n_iterations'] + 1)
    plot_y_per_x(x, data['mean_returns'], title='Returns', run_window=1, legend=True)
    plot_y_per_x(x, data['mean_fhts'], title='MFHT')

    # plot policy
    env = env.unwrapped
    if env.d <= 2:
        value_functions, means, stds = agent.eval_model_state_space(data, backup_iterations)

    if env.d == 1:
        colors, labels = get_colors_and_labels(backup_iterations, iter_str='Iter.')
        plot_det_policies_1d(env, means, sol_hjb.u_opt, colors=colors, labels=labels, loc='upper left')
        plot_ys_1d(env, value_functions, -sol_hjb.value_function)

    if env.d == 2:
        plot_det_policy_2d(env, means[-1].reshape(env.n_states_axis+(env.d,)), sol_hjb.u_opt)
        plot_value_function_2d(env, value_functions[-1].reshape(env.n_states_axis), sol_hjb.value_function)


if __name__ == "__main__":
    main()
