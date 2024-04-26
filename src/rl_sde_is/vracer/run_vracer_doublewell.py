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
    parser.description = 'Run V-racer for the sde importance sampling environment \
                          with a double well potential'
    args = parser.parse_args()

    # create gym environment
    gym_env = gym.make(
        'sde-is-doublewell-mgf-v0',
        beta=args.beta,
        reward_type=args.reward_type,
        baseline_scale_factor=args.baseline_scale_factor,
        state_init_dist=args.state_init_dist,
    )
    gym_env = RecordEpisodeStatistics(gym_env, args.n_episodes)


    # define Korali experiment 
    e = korali.Experiment()

    # define Problem Configuration
    set_korali_problem(e, gym_env, args)

    # Set V-RACER training parameters
    set_vracer_train_params(e, gym_env, args)

    # Define Variables
    e["Variables"][0]["Name"] = "Position"
    e["Variables"][0]["Type"] = "State"
    e["Variables"][1]["Name"] = "Control"
    e["Variables"][1]["Type"] = "Action"
    e["Variables"][1]["Lower Bound"] = -5.0
    e["Variables"][1]["Upper Bound"] = +5.0
    e["Variables"][1]["Initial Exploration Noise"] = 1.0

    # vracer
    data = vracer(e, gym_env, args, load=args.load)

    # plots
    if not args.plot: return

    # time step
    dt = gym_env.unwrapped.dt

    if args.beta == 1.0:
        plot_return_per_episode_std(data['returns'], ylim=(-10, 0))
        plot_fht_per_episode_std(dt*data['time_steps'], ylim=(0, 5))
        plot_psi_is_per_episode_std(np.exp(data['log_psi_is']), ylim=(1e-2, 5e-1))

    if args.beta == 4.0:
        xlim = (0, 2500)
        plot_return_per_episode_std(data['returns'], xlim=xlim, ylim=(-120, 0))
        plot_fht_per_episode_std(dt*data['time_steps'], xlim=xlim, ylim=(0, 100))
        plot_psi_is_per_episode_std(np.exp(data['log_psi_is']), xlim=xlim, ylim=(1e-4, 1e-1))

if __name__ == '__main__':
    main()
