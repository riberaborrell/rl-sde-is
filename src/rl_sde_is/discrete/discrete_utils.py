import torch
import torch.nn as nn

from rl_sde_is.utils.models import mlp

class DQNModel(nn.Module):

    def __init__(self, state_dim, n_actions, hidden_sizes, activation):
        super(DQNModel, self).__init__()
        self.sizes = [state_dim] + list(hidden_sizes) + [n_actions]
        self.q = mlp(self.sizes, activation)
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-4, 5e-4)
                nn.init.uniform_(module.bias, -5e-4, 5e-4)

    def forward(self, state):
        return self.q(state)

class DuelingCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_sizes, activation):
        super(DuelingCritic, self).__init__()

        self.n_actions = n_actions
        sizes = [state_dim] + list(hidden_sizes)
        self.net = mlp(sizes, activation, activation)
        self.value_layer = nn.Linear(hidden_sizes[-1], 1)
        self.advantage_layer = nn.Linear(hidden_sizes[-1], n_actions)

    def forward(self, state):
        y = self.net(state)
        value = self.value_layer(y)
        advantage = self.advantage_layer(y)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

