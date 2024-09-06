import json
from os import path

import numpy as np
import torch
import torch.nn as nn

from rl_sde_is.vracer.vracer_utils import get_vracer_rel_dir_path
from rl_sde_is.utils.models import mlp

def vracer_softplus_fn(x):
    return 0.5 * (x + torch.sqrt(x**2 + 1))

class OutputActivation(nn.Module):
    def __init__(self, idx, std_params_scaling):
        super().__init__()
        self.idx = idx
        self.std_output_activation = vracer_softplus_fn
        self.std_params_scaling = torch.tensor(std_params_scaling, dtype=torch.float32)

    def forward(self, x):
        if x.ndim == 1:
            x[self.idx] = self.std_output_activation(self.params_scaling*x[self.idx])
        elif x.ndim == 2:
            x[:, self.idx] = self.std_params_scaling * self.std_output_activation(x[:, self.idx])
            #x[:, self.idx] = self.std_output_activation((self.params_scaling*x)[:, self.idx])
        return x

class VracerModel(nn.Module):

    def __init__(self, d_in, d_out, hidden_sizes, activation, policy_params_scaling):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_out_tot = 1 + 2 * d_out # V, mus, stds
        self.sizes = [d_in] + list(hidden_sizes) + [self.d_out_tot]
        output_activation = OutputActivation(idx=slice(1+self.d_out, self.d_out_tot),
                                             std_params_scaling=policy_params_scaling[d_in:])
        self.model = mlp(self.sizes, activation, output_activation)

    def forward(self, state):
        return self.model.forward(state)

    def evaluate(self, state, idx):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            if state.ndim == 1:
                return self.forward(state)[idx].numpy()
            elif state.ndim == 2:
                return self.forward(state)[:, idx].numpy()
            elif state.ndim == 3:
                return self.forward(state)[:, :, idx].numpy()

    def value_function(self, state):
        return self.evaluate(state, slice(0, 1))

    def mean(self, state):
        return self.evaluate(state, slice(1, 1+self.d_out))

    def std(self, state):
        return self.evaluate(state, slice(1+self.d_out, self.d_out_tot))

def get_policy_hyperparameters(file: str):
    with open(file, "r") as f:
        dd = json.load(f)
    policy_params_scaling = dd["Solver"]["Policy"]["Parameter Scaling"]
    return policy_params_scaling


def get_model_hyperparameters(file: str):
    with open(file, "r") as f:
        dd = json.load(f)

    # get input and output dimensions
    d_in = dd["State Vector Size"]
    d_out = dd["Action Vector Size"]

    # get model parameters
    params = torch.tensor(dd["Policy Hyperparameters"], dtype=torch.float32).squeeze()

    # assume feed forward architecture
    # load activation functions and hidden layer sizes
    activations = []
    hidden_sizes = []
    net = dd["Neural Network"]
    for entry in net['Hidden Layers']:
        if entry["Type"] == "Layer/Linear":
            hidden_sizes.append(entry["Output Channels"])
        elif entry["Type"] == "Layer/Activation":
            if entry["Function"] == "Elementwise/Tanh":
                activations.append(nn.Tanh())
            else:
                raise NotImplementedError(f"not implemented activation function {function}")
        else:
            raise NotImplementedError(f"not implemented layer type {layer_type}")

    return d_in, d_out, hidden_sizes, activations, params

def get_model_hyperparameters_from_korali(korali_file: str):
    with open(korali_file, "r") as f:
        e = json.load(f)

    # get input and output dimensions
    d_in = e["Problem"]["State Vector Size"]
    d_out = e["Problem"]["Action Vector Size"]

    # get model parameters
    try:
        params = np.array(
            e["Solver"]["Training"]["Best Policy"]["Policy Hyperparameters"]["Policy"]
        )
    except KeyError:
        try:
            params = np.array(
                e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"][0]
            )
        except KeyError:
            params = np.array(e["Solver"]["Training"]["Best Policy"]["Policy"])
    params = torch.tensor(params, dtype=torch.float32)

    # assume feed forward architecture
    # load activation functions and hidden layer sizes
    activations = []
    hidden_sizes = []
    NN = e["Solver"]["Neural Network"]
    for entry in NN['Hidden Layers']:
        if entry["Type"] == "Layer/Linear":
            hidden_sizes.append(entry["Output Channels"])
        elif entry["Type"] == "Layer/Activation":
            if entry["Function"] == "Elementwise/Tanh":
                activations.append(nn.Tanh())
            else:
                raise NotImplementedError(f"not implemented activation function {function}")
        else:
            raise NotImplementedError(f"not implemented layer type {layer_type}")

    return d_in, d_out, hidden_sizes, activations, params

def load_model(file: str):

    # load policy hyperparameters
    vracer_file = path.dirname(file) + '/latest'
    policy_params_scaling = get_policy_hyperparameters(vracer_file)
    #model_params_scaling = [1.] + policy_params_scaling

    d_in, d_out, hidden_sizes, activations, params = get_model_hyperparameters(file)

    # load model
    model = VracerModel(d_in, d_out, hidden_sizes, nn.Tanh(), policy_params_scaling)
    state_dict = model.state_dict()

    # build state dict
    i, s = 0, 0
    for key in state_dict:
        if 'weight' in key:
            size = model.sizes[i] * model.sizes[i+1]
        elif 'bias' in key:
            size = model.sizes[i+1]
            i += 1
        state_dict[key] = params[s:s+size].reshape(model.state_dict()[key].shape)
        s += size

    model.load_state_dict(state_dict)

    return model

def get_means(env, args, episodes):
    results_dir = get_vracer_rel_dir_path(env, args)
    means = []

    for ep in episodes:

        # load model
        model = load_model(results_dir + '/model{:08d}.json'.format(ep))

        # append actions following policy
        means.append(model.mean(env.state_space_h))

    return means

def get_value_functions(env, args, episodes):
    results_dir = get_vracer_rel_dir_path(env, args)
    value_functions = []

    for ep in episodes:

        # load model
        model = load_model(results_dir + '/model{:08d}.json'.format(ep))

        # append actions following policy
        value_functions.append(model.value_function(env.state_space_h))

    return value_functions

def eval_model_state_space(env, args, episodes):
    results_dir = get_vracer_rel_dir_path(env, args)
    value_functions, means, stds = [], [], []
    for ep in episodes:

        # load model
        model = load_model(results_dir + '/model{:08d}.json'.format(ep))

        # append actions following policy
        value_functions.append(model.value_function(env.state_space_h).squeeze())
        means.append(model.mean(env.state_space_h))
        stds.append(model.std(env.state_space_h))

    return value_functions, means, stds

