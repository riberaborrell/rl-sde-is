import json
import os

import numpy as np

import korali
from gym_sde_is.utils.sde import compute_is_functional
from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl_sde_is.utils.config import DATA_ROOT_DIR
from rl_sde_is.utils.path import get_vracer_dir_path, load_data, save_data

def get_vracer_rel_dir_path(env, args):
    return os.path.join(
        os.path.relpath(DATA_ROOT_DIR),
        args.dir_path,
    )

def set_korali_problem(e, env, args):
    from rl_sde_is.vracer.korali_environment import env as korali_env

    # problem configuration
    e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
    e["Problem"]["Environment Function"] = lambda s : korali_env(s, env, args)
    #e["Problem"]["Actions Between Policy Updates"] = 1
    e["Solver"]["Type"] = "Agent / Continuous / VRACER"

    # set seed
    if args.seed is not None:
        e["Random Seed"] = args.seed

def set_vracer_train_params(e, env, args):

    # agent configuration 
    e["Solver"]["Mode"] = "Training"
    e["Solver"]["Episodes Per Generation"] = 1 #args.n_episodes
    e["Solver"]["Experiences Between Policy Updates"] = args.policy_freq
    e["Solver"]["Learning Rate"] = args.lr
    e["Solver"]["Discount Factor"] = args.gamma

    # set L2 regularization
    #e["Solver"]["L2 Regularization"]["Enabled"] = False
    #e["Solver"]["L2 Regularization"]["Importance"] = 0.0001

    # set mini batch
    e["Solver"]["Mini Batch"]["Size"] = args.batch_size

    # set Experience Replay, REFER and policy settings
    e["Solver"]["Experience Replay"]["Start Size"] = 4096 # (2**12)
    e["Solver"]["Experience Replay"]["Maximum Size"] = 262144 # (2**18)
    e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 0.0
    e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = args.cutoff_scale
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
    e["File Output"]["Path"] = get_vracer_rel_dir_path(env, args)


def set_vracer_variables_toy(e, env, args):
    for i in range(env.d):
        idx = i
        e["Variables"][idx]["Name"] = "Position x{:d}".format(i)
        e["Variables"][idx]["Type"] = "State"

    for i in range(env.d):
        idx = env.d + i
        e["Variables"][idx]["Name"] = "Control u{:d}".format(i)
        e["Variables"][idx]["Type"] = "Action"
        e["Variables"][idx]["Lower Bound"] = - args.action_limit
        e["Variables"][idx]["Upper Bound"] = + args.action_limit
        e["Variables"][idx]["Initial Exploration Noise"] = args.expl_noise_init

def set_vracer_variables_butane(e, env, args):
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


def set_vracer_eval_params(e, env, args):
    e["Solver"]["Mode"] = "Testing"
    e["Solver"]["Testing"]["Sample Ids"] = [i for i in range(args.n_episodes)]
    e["Console Output"]["Verbosity"] = "Detailed"
    e["File Output"]["Enabled"] = True
    e["File Output"]["Frequency"] = 1
    e["File Output"]["Path"] = get_vracer_rel_dir_path(env, args)

def collect_vracer_results(env):
    data = {}
    data['time_steps'] = env.lengths
    data['returns'] = env.returns
    #data['log_psi_is'] = env.log_psi_is
    data['is_functional'] = compute_is_functional(env.girs_stoch_int, env.running_rewards,
                                                  env.terminal_rewards)
    return data

def vracer(env, args, load=False):

    # get dir path
    args.dir_path = get_vracer_dir_path(
        env,
        gamma=args.gamma,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        action_limit=args.action_limit,
        expl_noise_init=args.expl_noise_init,
        baseline_scale_factor=args.baseline_scale_factor,
        policy_freq=args.policy_freq,
        cutoff_scale=args.cutoff_scale,
        batch_size=args.batch_size,
        lr=args.lr,
        n_episodes=args.n_episodes,
        seed=args.seed,

    )
    args.rel_dir_path = get_vracer_rel_dir_path(env, args)

    # load results
    if load:
        return load_data(args.dir_path)

    # record statistic wrapper
    env = RecordEpisodeStatistics(env, args.n_episodes)

    # define Korali experiment 
    e = korali.Experiment()

    # define Problem Configuration
    set_korali_problem(e, env, args)

    # set V-RACER training parameters
    set_vracer_train_params(e, env, args)

    # set V-RACER variables
    if 'butane' in env.name:
        set_vracer_variables_butane(e, env, args)
    else:
        set_vracer_variables_toy(e, env, args)

    # korali engine
    k = korali.Engine()

    # running Experiment
    k.run(e)

    # save results
    data = collect_vracer_results(env)
    save_data(data, args.dir_path)

    return data

"""
def get_timestamp(korali_file: str):
    with open(korali_file, "r") as f:
        e = json.load(f)
        timestamp = e["Timestamp"]
        #TODO parse it to datetime object
"""

