import json
import os

import korali
import numpy as np
from gym_sde_is.utils.sde import compute_is_functional

from rl_sde_is.utils.config import DATA_ROOT_DIR
from rl_sde_is.utils.path import load_data, save_data, get_dir_path

def get_vracer_params_str(args):
    if args.reward_type == 'baseline':
        baseline_str = 'baseline-factor{}_'.format(args.baseline_scale_factor)
    else:
        baseline_str = ''

    param_str = baseline_str \
              + 'expl-noise{:.1f}_'.format(args.expl_noise_init) \
              + 'policy-freq{:d}_'.format(args.policy_freq) \
              + 'n-episodes{:.0e}_'.format(args.n_episodes) \
              + 'seed{}'.format(args.seed)
    return param_str

def get_vracer_dir_path(gym_env, args):
    return get_dir_path(gym_env.unwrapped.__str__(), 'vracer', get_vracer_params_str(args))

def get_vracer_rel_dir_path(gym_env, args):
    return os.path.join(
        os.path.relpath(DATA_ROOT_DIR),
        gym_env.unwrapped.__str__(),
        'vracer',
        get_vracer_params_str(args),
    )

def set_korali_problem(e, gym_env, args):
    from rl_sde_is.vracer.korali_environment import env

    # problem configuration
    e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
    e["Problem"]["Environment Function"] = lambda s : env(s, gym_env, args)
    #e["Problem"]["Actions Between Policy Updates"] = 1
    e["Solver"]["Type"] = "Agent / Continuous / VRACER"

    # set seed
    if args.seed is not None:
        e["Random Seed"] = args.seed

def set_vracer_train_params(e, gym_env, args):

    # agent configuration 
    e["Solver"]["Mode"] = "Training"
    e["Solver"]["Episodes Per Generation"] = 1 #args.n_episodes
    e["Solver"]["Experiences Between Policy Updates"] = args.policy_freq
    e["Solver"]["Learning Rate"] = 0.0001
    e["Solver"]["Discount Factor"] = 1.0

    # set L2 regularization
    #e["Solver"]["L2 Regularization"]["Enabled"] = False
    #e["Solver"]["L2 Regularization"]["Importance"] = 0.0001

    # set mini batch
    e["Solver"]["Mini Batch"]["Size"] = 256

    # set Experience Replay, REFER and policy settings
    e["Solver"]["Experience Replay"]["Start Size"] = 4096 # (2**12)
    e["Solver"]["Experience Replay"]["Maximum Size"] = 262144 # (2**18)
    e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 0.0
    e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
    e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
    e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
    e["Solver"]["Experience Replay"]["Serialize"] = False

    # Set rescaling options
    #e["Solver"]["State Rescaling"]["Enabled"] = False
    #e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False

    # set policy type
    e["Solver"]["Policy"]["Distribution"] = "Normal"
    e["Solver"]["Neural Network"]["Engine"] = "OneDNN" # Intel
    #e["Solver"]["Neural Network"]["Engine"] = "cuDNN" # Nvidia
    e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

    # set neural network architecture
    for i in range(args.n_layers-1):

        # set linear layer
        e["Solver"]["Neural Network"]["Hidden Layers"][i*2]["Type"] = "Layer/Linear"
        e["Solver"]["Neural Network"]["Hidden Layers"][i*2]["Output Channels"] = args.d_hidden

        # set activation function
        e["Solver"]["Neural Network"]["Hidden Layers"][i*2 + 1]["Type"] = "Layer/Activation"
        e["Solver"]["Neural Network"]["Hidden Layers"][i*2 + 1]["Function"] = "Elementwise/Tanh"

    # set termination criteria
    e["Solver"]["Termination Criteria"]["Max Episodes"] = args.n_episodes
    #e["Solver"]["Termination Criteria"]["Max Experiences"] = args.n_total_steps

    # file output configuration
    e["Console Output"]["Verbosity"] = "Detailed"
    e["File Output"]["Enabled"] = True
    e["File Output"]["Frequency"] = args.backup_freq
    e["File Output"]["Path"] = get_vracer_rel_dir_path(gym_env, args)


def set_vracer_variables_toy(e, gym_env, args):
    for i in range(gym_env.d):
        idx = i
        e["Variables"][idx]["Name"] = "Position x{:d}".format(i)
        e["Variables"][idx]["Type"] = "State"

    for i in range(gym_env.d):
        idx = gym_env.d + i
        e["Variables"][idx]["Name"] = "Control u{:d}".format(i)
        e["Variables"][idx]["Type"] = "Action"
        e["Variables"][idx]["Lower Bound"] = - args.action_limit
        e["Variables"][idx]["Upper Bound"] = + args.action_limit
        e["Variables"][idx]["Initial Exploration Noise"] = args.expl_noise_init

def set_vracer_variables_butane(e, gym_env, args):
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
            e["Variables"][idx]["Lower Bound"] = - args.action_limit
            e["Variables"][idx]["Upper Bound"] = + args.action_limit
            e["Variables"][idx]["Initial Exploration Noise"] = args.expl_noise_init


def set_vracer_eval_params(e, gym_env, args):
    e["Solver"]["Mode"] = "Testing"
    e["Solver"]["Testing"]["Sample Ids"] = [i for i in range(args.n_episodes)]
    e["Console Output"]["Verbosity"] = "Detailed"
    e["File Output"]["Enabled"] = True
    e["File Output"]["Frequency"] = 1
    e["File Output"]["Path"] = get_vracer_rel_dir_path(gym_env, args)

def collect_vracer_results(gym_env):
    data = {}
    data['time_steps'] = gym_env.lengths
    data['returns'] = gym_env.returns
    #data['log_psi_is'] = gym_env.log_psi_is
    data['is_functional'] = compute_is_functional(gym_env.girs_stoch_int, gym_env.running_rewards,
                                                  gym_env.terminal_rewards)
    return data

def vracer(e, gym_env, args, load=False):

    # get dir path
    args.rel_dir_path = get_dir_path(
        gym_env.unwrapped.__str__(), 'vracer', get_vracer_params_str(args)
    )

    # load results
    if load:
        try:
            data = load_data(args.rel_dir_path)
            return data
        except FileNotFoundError as e:
            print(e)

    # korali engine
    k = korali.Engine()

    # Running Experiment
    k.run(e)

    # save results
    data = collect_vracer_results(gym_env)
    save_data(data, args.rel_dir_path)

    return data

"""
def get_timestamp(korali_file: str):
    with open(korali_file, "r") as f:
        e = json.load(f)
        timestamp = e["Timestamp"]
        #TODO parse it to datetime object
"""

