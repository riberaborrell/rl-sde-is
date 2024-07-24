import gymnasium as gym
import gym_sde_is
import numpy as np

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

from vracer_utils import *

def main():
    parser = get_base_parser()
    parser.description = 'Run V-racer for the importance sampling environment \
                          for the butane molecule'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-butane-{}-v0'.format(args.setting),
        temperature=args.temperature,
        gamma=10.0,
        T=args.T,
        state_init_dist=args.state_init_dist,
    )

    # vracer
    data = vracer(env, args, load=args.load)

    # plots
    if not args.plot: return

    # time step
    dt = env.unwrapped.dt

    plot_fht_per_episode_std(dt * data['time_steps'])
    plot_return_per_episode_std(data['returns'])
    plot_psi_is_per_episode_std(data['is_functional'])

if __name__ == '__main__':
    main()
