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
    parser.description = 'Run V-racer for the sde importance sampling environment \
                          with a toy potential.'
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-{}-{}-v0'.format(args.problem, args.setting),
        beta=args.beta,
        alpha=np.array(args.alpha),
        T=args.T,
        reward_type=args.reward_type,
        baseline_scale_factor=args.baseline_scale_factor,
        state_init_dist=args.state_init_dist,
    )
    env = RecordEpisodeStatistics(env, args.n_episodes)

    # define Korali experiment 
    e = korali.Experiment()

    # define Problem Configuration
    set_korali_problem(e, env, args)

    # Set V-RACER training parameters
    set_vracer_train_params(e, env, args)
    set_vracer_variables_toy(e, env, args)

    # vracer
    data = vracer(e, env, args, load=args.load)

    # plots
    if not args.plot: return

    # time step
    dt = env.unwrapped.dt
    plot_return_per_episode_std(data['returns'])
    plot_fht_per_episode_std(dt*data['time_steps'])
    plot_psi_is_per_episode_std(data['is_functional'])

if __name__ == '__main__':
    main()
