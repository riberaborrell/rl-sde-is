import json
import numpy as np
import torch
import torch.nn as nn

from rl_sde_is.vracer.vracer_utils import get_vracer_rel_dir_path
from rl_sde_is.models import mlp

def square_plus(x, b=1):
    return 0.5 * (x + torch.sqrt(x**2 + b))

class OutputActivation(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        #self.stds_output_activation = square_plus
        self.stds_output_activation = nn.Softplus()

    def forward(self, x):
        if x.ndim == 1:
            x[self.idx] = self.stds_output_activation(x[self.idx])
        elif x.ndim == 2:
            x[:, self.idx] = self.stds_output_activation(x[:, self.idx])
        return x


class VracerModel(nn.Module):

    def __init__(self, d_in, d_out, hidden_sizes, activation):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_out_tot = 1 + 2 * d_out # V, mus, stds
        self.sizes = [d_in] + list(hidden_sizes) + [self.d_out_tot]
        output_activation = OutputActivation(slice(1+self.d_out, self.d_out_tot))
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

    def policy(self, state):
        return self.evaluate(state, slice(1, 1+self.d_out))

    def stds(self, state):
        return self.evaluate(state, slice(1+self.d_out, self.d_out_tot))


def load_model(korali_file: str):
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

    # load model
    model = VracerModel(d_in, d_out, hidden_sizes, nn.Tanh())
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

    # load params
    model.load_state_dict(state_dict)

    return model

def get_policies(env, args, episodes):
    results_dir = get_vracer_rel_dir_path(env, args)
    policies = []

    for ep in episodes:

        # load model
        model = load_model(results_dir + '/gen{:08d}.json'.format(ep))

        # append actions following policy
        policies.append(model.policy(env.state_space_h))

    return policies

def get_value_functions(env, args, episodes):
    results_dir = get_vracer_rel_dir_path(env, args)
    value_functions = []

    for ep in episodes:

        # load model
        model = load_model(results_dir + '/gen{:08d}.json'.format(ep))

        # append actions following policy
        value_functions.append(model.value_function(env.state_space_h))

    return value_functions

def eval_model_state_space(env, args, episodes):
    results_dir = get_vracer_rel_dir_path(env, args)
    value_functions, policies, stds = [], [], []
    for ep in episodes:

        # load model
        model = load_model(results_dir + '/gen{:08d}.json'.format(ep))

        # append actions following policy
        policies.append(model.policy(env.state_space_h))
        value_functions.append(model.value_function(env.state_space_h).squeeze())
        stds.append(model.stds(env.state_space_h))

    return policies, value_functions, stds

