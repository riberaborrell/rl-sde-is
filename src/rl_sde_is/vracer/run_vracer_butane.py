import gymnasium as gym
import gym_sde_is
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics
import numpy as np
import korali

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

from vracer_utils import *

def main():
    parser = get_base_parser()
    parser.description = 'Run V-racer for the importance sampling environment \
                          for the butane molecule'
    args = parser.parse_args()

    # create gym environment
    gym_env = gym.make(
        'sde-is-butane-{}-v0'.format(args.setting),
        temperature=args.temperature,
        gamma=10.0,
        T=args.T,
    )
    gym_env = RecordEpisodeStatistics(gym_env, args.n_episodes)

    # define Korali experiment 
    e = korali.Experiment()

    # Defining Problem Configuration
    set_korali_problem(e, gym_env, args)

    # Set V-RACER training parameters
    args.n_layers = 4
    args.d_hidden = 64
    args.action_limit = 1.0
    set_vracer_train_params(e, gym_env, args)
    set_vracer_variables_butane(e, gym_env, args)

    # vracer
    data = vracer(e, gym_env, args, load=args.load)

    # plots
    if not args.plot: return

    # time step
    dt = gym_env.unwrapped.dt

    plot_fht_per_episode_std(dt * data['time_steps'])
    plot_return_per_episode_std(data['returns'])
    plot_psi_is_per_episode_std(data['is_functional'])

if __name__ == '__main__':
    main()
