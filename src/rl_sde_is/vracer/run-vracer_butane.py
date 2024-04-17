import gymnasium as gym
import gym_sde_is
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics
import numpy as np
import korali

from rl_sde_is.utils.base_parser import get_base_parser
from rl_sde_is.utils.plots import *

from vracer_utils import set_korali_problem, set_vracer_train_params, vracer

def main():
    parser = get_base_parser()
    parser.description = 'Run V-racer for the importance sampling environment \
                          for the butane molecule'
    args = parser.parse_args()

    # create gym environment
    gym_env = gym.make(
        'sde-is-butane-mgf-v0',
        temperature=args.temperature,
        gamma=10.0,
    )
    gym_env = RecordEpisodeStatistics(gym_env, deque_size=int(args.n_episodes))

    # define Korali experiment 
    e = korali.Experiment()

    # Defining Problem Configuration
    set_korali_problem(e, gym_env, args)

    # Set V-RACER training parameters
    set_vracer_train_params(e, gym_env, args)

    # Define Variables
    for i in range(4):
        for j in range(3):
            idx = i*3+j
            e["Variables"][idx]["Name"] = "Position (C{:d} x{:d}-axis)".format(i, j)
            e["Variables"][idx]["Type"] = "State"

    for i in range(4):
        for j in range(3):
            idx = 12 + i*3+j
            e["Variables"][idx]["Name"] = "Control ({:d}-{:d})".format(i, j)
            e["Variables"][idx]["Type"] = "Action"
            e["Variables"][idx]["Lower Bound"] = -1.0
            e["Variables"][idx]["Upper Bound"] = +1.0
            e["Variables"][idx]["Initial Exploration Noise"] = 1.0

    # vracer
    data = vracer(e, gym_env, args, load=args.load)

    if args.plot:
        plot_returns_std_episodes(data['returns'])
        plot_time_steps_std_episodes(data['time_steps'])
        plot_psi_is_std_episodes(data['psi_is'])


if __name__ == '__main__':
    main()
