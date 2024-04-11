import gymnasium as gym
import gym_sde_is
import korali

from base_parser import get_base_parser
from config import set_korali_problem, set_vracer_train_params

def main():
    parser = get_base_parser()
    parser.description = 'Run V-racer for the importance sampling environment \
                          for the butane molecule'
    args = parser.parse_args()

    # create gym environment
    gym_env = gym.make(
        'sde-is-butane-mgf-v0',
        temperature=600.0,
        gamma=10.0,
    )

    # define Korali Problem
    k = korali.Engine()
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
            e["Variables"][idx]["Initial Exploration Noise"] = 0.1

    # Running Experiment
    k.run(e)

if __name__ == '__main__':
    main()
