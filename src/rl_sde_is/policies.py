import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#from models import FeedForwardNN
# pi constant as a tensor
pi = Variable(torch.FloatTensor([math.pi])).cpu()

class DiscreteStochPolicy(nn.Module):
    def __init__(self, d_in, hidden_size, d_out, activation=torch.tanh):
        super(DiscreteStochPolicy, self).__init__()

        # input, hidden and output dimensions
        self.d_in = d_in
        self.hidden_size = hidden_size
        self.d_out = d_out

        # two linear layers
        self.linear1 = nn.Linear(d_in, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, d_out)

        # activation function
        self.activation = activation

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        action_scores = self.linear3(x)
        return F.softmax(action_scores, dim=-1)

class GaussStochPolicy1(nn.Module):
    def __init__(self, d_in, hidden_size, d_out, activation=torch.tanh, seed=0):
        super(GaussStochPolicy1, self).__init__()

        # fix seed
        torch.manual_seed(seed)

        # input, hidden and output dimensions
        self.d_in = d_in
        self.hidden_size = hidden_size
        self.d_out = d_out

        # mean and sigma share the same first layer
        self.linear1 = nn.Linear(d_in, hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_out)
        self.linear2_ = nn.Linear(hidden_size, d_out)

        # activation function
        self.activation = activation

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.activation(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = F.softplus(self.linear2_(x))

        return mu, sigma_sq

    def sample_action(self, state):
        # get parameters of the policy
        mu, sigma_sq = self.forward(state)

        # normal sampled centered at mu
        eps = torch.randn(mu.size())

        # return normal sampled  action
        return (mu + sigma_sq.sqrt() * Variable(eps).cpu()).data


    def probability(self, state, action):

        # get parameters of the policy
        mu, sigma_sq = self.forward(state)

        # gaussian function (exponent)
        a = (-1 * (Variable(action) - mu).pow(2) / (2*sigma_sq)).exp()

        # gaussian function (normalization factor)
        b = 1 / (2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()

        return a * b

class GaussStochPolicy2(nn.Module):
    def __init__(self, d_in, hidden_size, d_out, activation=torch.tanh, seed=0):
        super(GaussStochPolicy2, self).__init__()

        # fix seed
        torch.manual_seed(seed)

        # input, hidden and output dimensions
        self.d_in = d_in
        self.hidden_size = hidden_size
        self.d_out = d_out
        self.sigma_sq = 0.1

        # mean and sigma have different parametrizations
        self.mu_linear1 = nn.Linear(d_in, hidden_size)
        self.mu_linear2 = nn.Linear(hidden_size, d_out)
        #self.sigma_linear1 = nn.Linear(d_in, hidden_size)
        #self.sigma_linear2 = nn.Linear(hidden_size, d_out)

        # activation function
        self.activation = activation

    def forward(self, x):
        x = torch.FloatTensor(x)

        mu = self.activation(self.mu_linear1(x))
        mu = self.mu_linear2(mu)

        #sigma_sq = self.activation(self.sigma_linear1(inputs))
        #sigma_sq = F.softplus(self.sigma_linear2(sigma_sq))

        return mu, self.sigma_sq * torch.ones_like(x)

    def sample_action(self, state):
        # get parameters of the policy
        mu, sigma_sq = self.forward(state)

        # normal sampled centered at mu
        eps = torch.randn(mu.size())

        # return normal sampled  action
        return (mu + sigma_sq.sqrt() * Variable(eps).cpu()).data


    def probability(self, state, action):

        # get parameters of the policy
        mu, sigma_sq = self.forward(state)

        # gaussian function (exponent)
        action = torch.FloatTensor(action)
        a = (-1 * (Variable(action) - mu).pow(2) / (2*sigma_sq)).exp()

        # gaussian function (normalization factor)
        b = 1 / (2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()

        return a * b

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        y = F.relu(self.linear1(state))
        y = F.relu(self.linear2(y))
        y = self.max_action * torch.tanh(self.linear3(y))
        return y
