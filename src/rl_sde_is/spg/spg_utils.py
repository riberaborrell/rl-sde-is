import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, MultivariateNormal

from rl_sde_is.models import mlp

class StochasticPolicy(nn.Module):
    def distribution(self, state):
        raise NotImplementedError

    def log_prob_from_distribution(self, dist, action):
        raise NotImplementedError

    def sample_action(self, state, log_prob=False):
        ''' Returns an action sampled from the policy distribution for given state.
            Optionally computes the log likelihood of the action under the distribution.
        '''
        with torch.no_grad():
            dist = self.distribution(state)
            action = dist.sample()
            if log_prob:
                log_prob_action = self.log_prob_from_distribution(dist, action).numpy()
            else:
                log_prob_action = None
            return action.numpy(), log_prob_action

    def forward(self, state, action=None):
        '''returns the corresponding policy action distribution for given state.
           Optionally computes the log likelihood of given action under that distribution
        '''
        dist = self.distribution(state)
        log_prob = None
        if action is not None:
            log_prob = self.log_prob_from_distribution(dist, action)
        return dist, log_prob

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                nn.init.uniform_(module.weight, -5e-3, 5e-3)
                nn.init.uniform_(module.bias, -5e-3, 5e-3)

class CategoricalPolicy(StochasticPolicy):
    def __init__(self, state_dim, n_actions, hidden_sizes, activation, seed=0):
        super().__init__()

        # fix seed
        #torch.manual_seed(seed)

        self.sizes = [state_dim] + list(hidden_sizes) + [n_actions]
        self.probs = mlp(self.sizes, activation, nn.Softmax(dim=-1))
        #self.logits = mlp(self.sizes, activation)

    def distribution(self, state):
        probs = self.probs(state)
        return Categorical(probs=probs)
        #logits = self.logits(state)
        #return Categorical(logits=logits)

    def log_prob_from_distribution(self, dist, action):
        return dist.log_prob(action)

class GaussianPolicy(StochasticPolicy):
    '''Gaussian Policy
    '''

    def mean_and_cov(self, state):
        raise NotImplementedError

    def distribution(self, state):
        mean, cov = self.mean_and_cov(state)
        return Normal(mean, cov)

    def log_prob_from_distribution(self, dist, action):
        return dist.log_prob(action).sum(axis=-1)


class GaussianPolicyConstantCov(GaussianPolicy):
    '''Gaussian Policy with constant covariance matrix
    '''
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, sigma, seed=0):
        super().__init__()

        # fix seed
        #torch.manual_seed(seed)

        # mean nn
        self.sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.mean = mlp(self.sizes, activation)
        self.apply(self.init_last_layer_weights)

        # constant covariance matrix
        self.sigma = sigma
        self.cov = sigma * torch.ones(action_dim)

    def mean_and_cov(self, state):
        return self.mean.forward(state), self.cov


class GaussianPolicyLearntCov(GaussianPolicy):
    '''Gaussian Policy with learnt covariance matrix
    '''
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, seed=0):
        super().__init__()

        # fix seed
        #torch.manual_seed(seed)

        # mean nn
        sizes = [state_dim] + list(hidden_sizes)
        self.net = mlp(sizes, activation, activation)
        self.mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_cov = nn.Linear(hidden_sizes[-1], action_dim)
        #self.cov = F.softplus(nn.Linear(hidden_sizes[-1], action_dim))
        #self.act_limit = act_limit

    def mean_and_cov(self, state):
        y = self.net.forward(state)
        mean = self.mean.forward(y)
        log_cov = self.log_cov.forward(y)
        return mean, torch.exp(log_cov)


class QValueFunction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.sizes = [state_dim + action_dim] + list(hidden_sizes) + [1]
        self.q = mlp(self.sizes, activation)

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return torch.squeeze(q, axis=-1)
