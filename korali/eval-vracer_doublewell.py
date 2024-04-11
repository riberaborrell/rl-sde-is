import os

import gymnasium as gym
import gym_sde_is
import korali

from base_parser import get_base_parser
from config import set_korali_problem, set_vracer_eval_params

def main():

    ### Parsing arguments
    parser = get_base_parser()
    parser.description = 'Run V-racer for the sde importance sampling environment with a double well potential'
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
        help='Set inverse of the temperature. Default: 1.',
    )
    args = parser.parse_args()

    ### create gym environment
    gym_env = gym.make(
        'sde-is-doublewell-mgf-v0',
        beta=args.beta,
        reward_type=args.reward_type,
        state_init_dist=args.state_init_dist,
    )

    ### Defining Korali Problem
    k = korali.Engine()
    e = korali.Experiment()

    ### Defining Problem Configuration
    set_korali_problem(e, gym_env, args)

    ### Define results directory and loading results
    results_dir = os.path.join('results', gym_env.unwrapped.__str__(), 'vracer')

    found = e.loadState(results_dir + '/latest')
    if found == True:
        print("[Korali] Evaluating previous run...\n");
    else:
        print("[Korali] Error: could not find results in folder: " + results_dir)
        exit(-1)

    ### Set V-RACER evaluation parameters
    set_vracer_eval_params(e, gym_env, args)

    ### Run Experiment
    k.run(e)

if __name__ == '__main__':
    main()
