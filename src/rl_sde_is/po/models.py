import numpy as np
import torch
import torch.nn as nn

from rl_sde_is.utils.models import mlp
from rl_sde_is.spg.spg_utils import GaussianPolicyLearntCov

class ActorCriticModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, std_init=1.0, seed=None):
        super().__init__()

        # fix seed
        if seed is not None:
            torch.manual_seed(seed)

        # model for the value function 
        self.critic_sizes = [state_dim] + list(hidden_sizes) + [1]
        self.critic = mlp(
            sizes=self.critic_sizes,
            activation=activation,
            output_activation=nn.Identity(),
        )

        # model for a gaussian stochastic policy
        self.actor = GaussianPolicyLearntCov(
            state_dim, action_dim, hidden_sizes,
            activation=activation, std_init=std_init,
        )

    def get_value(self, state):
        with torch.no_grad():
            return self.critic(state).numpy()

    def sample_action(self, state, log_prob=False):
        return self.actor.sample_action(state, log_prob=log_prob)

    def get_action_and_value(self, state, action=None):
        value = self.get_value(state)
        action, log_prob_action = self.actor.sample_action(state, log_prob=True)
        return action, log_prob_action, value

