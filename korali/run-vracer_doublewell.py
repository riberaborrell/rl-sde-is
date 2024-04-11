import gymnasium as gym
import gym_sde_is
import korali

from base_parser import get_base_parser
from config import set_korali_problem, set_vracer_train_params

def main():
    parser = get_base_parser()
    parser.description = 'Run V-racer for the sde importance sampling environment \
                          with a double well potential'
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
        help='Set inverse of the temperature. Default: 1.',
    )
    args = parser.parse_args()

    # create gym environment
    gym_env = gym.make(
        'sde-is-doublewell-mgf-v0',
        beta=args.beta,
        reward_type=args.reward_type,
        state_init_dist=args.state_init_dist,
    )

    # define Korali Problem
    k = korali.Engine()
    e = korali.Experiment()

    # Defining Problem Configuration
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
    e["Variables"][1]["Initial Exploration Noise"] = 0.3

    # Running Experiment
    k.run(e)

if __name__ == '__main__':
    main()
