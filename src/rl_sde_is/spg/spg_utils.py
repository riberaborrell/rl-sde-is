import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from rl_sde_is.utils.models import mlp

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

    def mean_and_std(self, state):
        raise NotImplementedError

    def mean(self, state):
        raise NotImplementedError

    def std(self, state):
        raise NotImplementedError

    def distribution(self, state):
        mean, std = self.mean_and_std(state)
        return Normal(mean, std)

    def log_prob_from_distribution(self, dist, action):
        return dist.log_prob(action).sum(axis=-1)

class GaussianPolicyConstantCov(GaussianPolicy):
    '''Gaussian Policy with constant covariance matrix
    '''
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, std, seed=0):
        super().__init__()

        # fix seed
        #torch.manual_seed(seed)

        # mean nn
        self.sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.mean_fn = mlp(self.sizes, activation)
        self.apply(self.init_last_layer_weights)

        # constant covariance matrix
        self.std_const = torch.ones(action_dim) * std

    def mean_and_std(self, state):
        std = torch.empty_like(state)
        std[:] = self.std_const
        return self.mean_fn.forward(state), std

    def mean(self, state):
        return self.mean_fn.forward(state)

    def std(self, state):
        std = torch.empty_like(state)
        std[:] = self.std_const
        return std

def vracer_softplus_fn(x):
    return x + torch.sqrt(x**2 + 1)

class GaussianPolicyLearntCov(GaussianPolicy):
    '''Gaussian Policy with learnt covariance matrix
    '''
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, std_init=1.0, seed=0):
        super().__init__()

        # fix seed
        #torch.manual_seed(seed)

        # mean nn
        sizes = [state_dim] + list(hidden_sizes)
        self.net = mlp(sizes, activation, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.std_init = std_init

        # initialize mean weights to "zero"
        nn.init.uniform_(self.mean_layer.weight, -5e-3, 5e-3)
        nn.init.uniform_(self.mean_layer.bias, -5e-3, 5e-3)

        # initialize std weights to "zero"
        nn.init.uniform_(self.std_layer.weight, -5e-3, 5e-3)
        nn.init.uniform_(self.std_layer.bias, -5e-3, 5e-3)

    def mean_and_std(self, state):
        y = self.net.forward(state)
        mean = self.mean_layer.forward(y)
        z = self.std_layer.forward(y)
        std = self.std_init * vracer_softplus_fn(z)
        return mean, std

    def mean(self, state):
        y = self.net.forward(state)
        return self.mean_layer.forward(y)

    def std(self, state):
        y = self.net.forward(state)
        z = self.std_layer.forward(y)
        return self.std_init * vracer_softplus_fn(z)

def compute_importance_weights(action, behav_mean, behav_std, curr_mean, curr_std):
    log_behav_policy = -0.5 * ((action - behav_mean) / behav_std)**2 - torch.log(behav_std)
    log_curr_policy = -0.5 * ((action - curr_mean) / curr_std)**2 - torch.log(curr_std)
    return torch.exp(log_behav_policy - log_curr_policy)
