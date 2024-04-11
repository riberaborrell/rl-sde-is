import os

from environment import environment as env

def set_korali_problem(e, gym_env, args):

    ### Defining Problem Configuration
    e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
    e["Problem"]["Environment Function"] = lambda s : env(s, gym_env, args)
    e["Problem"]["Actions Between Policy Updates"] = 1
    e["Solver"]["Type"] = "Agent / Continuous / VRACER"

def set_vracer_train_params(e, gym_env, args):

    ### Defining Agent Configuration 
    e["Solver"]["Mode"] = "Training"
    e["Solver"]["Episodes Per Generation"] = 1
    e["Solver"]["Experiences Between Policy Updates"] = 1
    e["Solver"]["Learning Rate"] = 0.0001
    e["Solver"]["Discount Factor"] = 1.0 # try with 0.995
    e["Solver"]["Mini Batch"]["Size"] = 256

    ### Set Experience Replay, REFER and policy settings
    e["Solver"]["Experience Replay"]["Start Size"] = 4096 # 4096 (2**12)
    e["Solver"]["Experience Replay"]["Maximum Size"] = 262144 # 2**18
    e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
    e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0
    e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
    e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
    e["Solver"]["Experience Replay"]["Serialize"] = True

    # Set rescaling options
    #e["Solver"]["State Rescaling"]["Enabled"] = True
    #e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True

    ### Set policy type
    e["Solver"]["Policy"]["Distribution"] = args.distribution
    e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
    e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
    #e["Solver"]["L2 Regularization"]["Enabled"] = True
    #e["Solver"]["L2 Regularization"]["Importance"] = 1.0

    ### set neural network architecture
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = args.d_hidden
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = args.d_hidden
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

    ### Defining Termination Criteria
    e["Solver"]["Termination Criteria"]["Max Experiences"] = args.max_experiences

    ### Setting file output configuration
    e["Console Output"]["Verbosity"] = "Detailed"
    e["File Output"]["Enabled"] = True
    e["File Output"]["Frequency"] = 500
    e["File Output"]["Path"] = os.path.join('results', gym_env.unwrapped.__str__(), 'vracer')

def set_vracer_eval_params(e, gym_env, args):
    e["Solver"]["Mode"] = "Testing"
    e["Solver"]["Testing"]["Sample Ids"] = [i for i in range(args.n_episodes)]
    e["Console Output"]["Verbosity"] = "Detailed"
    e["File Output"]["Enabled"] = True
    e["File Output"]["Frequency"] = 1
    e["File Output"]["Path"] = os.path.join('results', gym_env.unwrapped.__str__(), 'vracer')
