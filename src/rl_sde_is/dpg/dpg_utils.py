import torch
import torch.nn as nn

from rl_sde_is.models import mlp

class DeterministicPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.policy = mlp(sizes=self.sizes, activation=activation)
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-3, 5e-3)
                nn.init.uniform_(module.bias, -5e-3, 5e-3)

    def forward(self, state):
        return self.policy.forward(state)

class QValueFunction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim + action_dim] + list(hidden_sizes) + [1]
        self.q = mlp(self.sizes, activation)
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-4, 5e-4)
                nn.init.uniform_(module.bias, -5e-4, 5e-4)

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return torch.squeeze(q, axis=-1)



class ValueFunction(nn.Module):

    def __init__(self, state_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim] + list(hidden_sizes) + [1]
        self.v = mlp(self.sizes, activation)

    def forward(self, state):
        return self.v.forward(state)

class AValueFunction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim + action_dim] + list(hidden_sizes) + [1]
        self.a = mlp(self.sizes, activation)

    def forward(self, state, action):
        a = self.a(torch.cat([state, action], dim=-1))
        return torch.squeeze(a, axis=-1)


