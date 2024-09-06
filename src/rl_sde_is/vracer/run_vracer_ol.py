import gymnasium as gym
import gym_sde_is
import numpy as np

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

from vracer_utils import vracer

def main():
    parser = get_base_parser()
    parser.description = 'Run V-racer for the sde importance sampling environment \
                          with a toy potential.'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        dt=args.dt,
        beta=args.beta,
        alpha=args.alpha,
        T=args.T,
        reward_type=args.reward_type,
        baseline_scale_factor=args.baseline_scale_factor,
        state_init_dist=args.state_init_dist,
        n_steps_lim=args.n_steps_lim,
    )

    # run vracer
    data = vracer(env, args, load=args.load)

    # plots
    if not args.plot: return

    # returns, mfhts, and is functional
    x = np.arange(args.n_episodes)
    dt = env.unwrapped.dt
    plot_y_per_episode(x, data['returns'], title='Objective function', run_window=10)
    plot_y_per_episode(x, dt*data['time_steps'], title='MFHT', run_window=10)
    plot_y_per_episode(x, data['is_functional'], title='$\Psi(s_0)$', run_window=10)


if __name__ == '__main__':
    main()
