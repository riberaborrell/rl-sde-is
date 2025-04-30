import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
from gym_sde_is.wrappers.save_episode_trajectory import SaveEpisodeTrajectoryVect

from rl_sde_is.spg.spg_utils import GaussianPolicyConstantCov, GaussianPolicyLearntCov
from rl_sde_is.spg.replay_memories import ReplayMemoryReturn as Memory
from rl_sde_is.utils.approximate_methods import evaluate_stoch_policy_model, \
                                                train_stochastic_policy_from_hjb
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.numeric import cumsum_numpy as cumsum, normalize_array
from rl_sde_is.utils.path import get_reinforce_stoch_dir_path, load_data, save_data, save_model, load_model
from rl_sde_is.utils.plots import initialize_gaussian_policy_1d_figure, update_gaussian_policy_1d_figure


class ReinforceStochastic:
    def __init__(self, env, gamma=1.0, expectation_type='random-time',
                 return_type='initial-return', estimate_z=True, policy_type='learnt-cov',
                 policy_noise=1.0, theta_init='null', n_layers=2, d_hidden_layer=32,
                 optim_type='sgd', batch_size=100,
                 mini_batch_size_type='adaptive', mini_batch_size=1, lr=1e-2, learn_value=False,
                 lr_value=None, n_grad_iterations=100, seed=None, norm_returns=True, cuda=False):

        if expectation_type == 'on-policy' and mini_batch_size is None:
            raise ValueError('The mini_batch_size must be provided when using on-policy')

        # agent
        self.agent = 'reinforce-stoch-{}'.format(expectation_type)

        # environment
        self.env = env

        # discount
        self.gamma = gamma

        # get state and action dimensions
        self.state_dim = env.unwrapped.d
        self.action_dim = env.unwrapped.d

        # expectation type and return type
        self.expectation_type = expectation_type
        self.return_type = return_type

        # state space (on-policy) expectation
        if expectation_type == 'on-policy':
            self.estimate_z = estimate_z
            self.mini_batch_size = mini_batch_size
            self.mini_batch_size_type = mini_batch_size_type

        # normalize returns
        self.norm_returns = norm_returns

        # stochastic policy
        self.policy_type = policy_type
        self.policy_noise = policy_noise
        self.theta_init = theta_init
        self.n_layers = n_layers
        self.d_hidden_layer = d_hidden_layer

        # stohastic gradient descent
        self.optim_type = optim_type
        self.batch_size = batch_size
        self.lr = lr
        self.learn_value = learn_value
        if learn_value:
            self.lr_value = lr_value
        self.n_grad_iterations = n_grad_iterations

        # cuda device
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

        # policy model
        hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
        if policy_type == 'const-cov':
            self.policy = GaussianPolicyConstantCov(state_dim=env.unwrapped.d, action_dim=env.unwrapped.d,
                                               hidden_sizes=hidden_sizes, activation=nn.Tanh(),
                                               std=policy_noise).to(self.device)
        else:
            self.policy = GaussianPolicyLearntCov(
                state_dim=env.unwrapped.d, action_dim=env.unwrapped.d, hidden_sizes=hidden_sizes,
                activation=nn.Tanh(), std_init=policy_noise,
            ).to(self.device)

        # initialize value function model
        if learn_value:
            self.value = ValueFunction(
                state_dim=env.unwrapped.d, hidden_sizes=d_hidden_layers, activation=nn.Tanh()
            ).to(self.device)
        else:
            self.value = None

        # seed
        self.seed = seed

    def sample_trajectories(self):

        # initialization
        state, _ = self.env.reset(options={'batch_size': self.batch_size})

        # terminal state flag
        done = np.full((self.batch_size,), False)
        while not done.all():

            # sample action
            state_torch = torch.FloatTensor(state)
            action, _ = self.policy.sample_action(state_torch)

            # env step
            state, _, _, truncated, _ = self.env.step_vect(action)
            done = np.logical_or(self.env.unwrapped.been_terminated, truncated)

        # compute returns
        returns = []
        for i in range(self.batch_size):

            # compute initial returns
            if self.return_type == 'initial-return':
                returns.append(
                    np.full(self.env.get_wrapper_attr('lengths')[i], self.env.get_wrapper_attr('returns')[i])
                )

            # compute n-step returns
            else: # return_type == 'n-return'
                returns.append(cumsum(self.env.trajs_rewards[i]))

        return np.vstack(self.env.trajs_states), np.vstack(self.env.trajs_actions), np.hstack(returns)

    #TODO: try if normalizing the returns helps to reduce the variance
    # returns = normalize_advs_trick(returns)

    def sample_loss_random_time(self):
        ''' Sample and compute loss function corresponding to the policy gradient with
            random time expectation. Also update the policy parameters.
        '''

        # sample trajectories
        states, actions, returns = self.sample_trajectories()

        # normalize n-returns
        returns = normalize_array(returns, eps=1e-5)

        # convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        returns = torch.FloatTensor(returns)

        # compute log probs
        _, log_probs = self.policy.forward(states, actions)

        # calculate loss
        phi = - log_probs * returns

        # loss and loss variance
        loss = phi.sum() / self.batch_size
        with torch.no_grad():
            loss_var = phi.var().numpy()

        # reset gradients, compute gradients and update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, loss_var


    def sample_loss_on_policy(self):

        # sample trajectories
        states, actions, returns = self.sample_trajectories()

        # initialize memory
        memory = Memory(size=states.shape[0]+1, state_dim=self.state_dim, action_dim=self.action_dim)

        # store experiences in memory
        memory.store_vectorized(states, actions, returns=returns)

        # sample batch of experiences from memory
        if self.mini_batch_size_type == 'adaptive':
            mini_batch_size = round(memory.size / self.mini_batch_size)
        else:
            mini_batch_size = self.mini_batch_size
        batch = memory.sample_batch(mini_batch_size)
        _, log_probs = self.policy.forward(batch['states'], batch['actions'])

        # estimate mean trajectory length
        mean_length = self.env.get_wrapper_attr('lengths').mean() if self.estimate_z else 1

        # normalize n-returns
        returns = normalize_array(batch['returns'], eps=1e-5)

        # calculate loss
        phi = - (log_probs * returns)
        loss = phi.mean()
        with torch.no_grad():
            loss_var = phi.var().numpy()

        # reset and compute actor gradients
        self.optimizer.zero_grad()
        loss.backward()

        # scale gradients before updating parameters
        if self.estimate_z:
            with torch.no_grad():
                for param in self.policy.parameters():
                    if param.grad is not None:
                        param.grad *= mean_length

        # scale learning rate
        #self.optimizer.param_groups[0]['lr'] *= mean_length

        #update parameters
        self.optimizer.step()

        # re-scale learning rate back
        #optimizer.param_groups[0]['lr'] /= mean_length

        return loss, loss_var

    def run_agent(self, backup_freq=None, live_plot_freq=None, log_freq=100,
                  policy_opt=None, value_function_opt=None, load=False):

        # get dir path
        self.dir_path = get_reinforce_stoch_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(self.dir_path)

        # set seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # vectorized environment
        self.env = RecordEpisodeStatisticsVect(self.env, self.batch_size)

        # save states, action and rewards
        if self.return_type == 'n-return':
            self.env = SaveEpisodeTrajectoryVect(self.env, self.batch_size, track_rewards=True)

        # save states and actions 
        else: #self.return_type == 'initial-return':
            self.env = SaveEpisodeTrajectoryVect(self.env, self.batch_size)


        # define optimizer
        if self.optim_type == 'adam':
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        elif self.optim_type == 'sgd':
            self.optimizer = optim.SGD(self.policy.parameters(), lr=self.lr)
        else:
            raise ValueError('The optimizer {optim} is not implemented')

        if self.learn_value:
            self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr_value)

        # train params to fit hjb solution
        if self.theta_init == 'hjb':
            train_stochastic_policy_from_hjb(self.env, self.policy, policy_opt, load=True)

        # save algorithm parameters
        excluded = ['env', 'policy', 'value', 'optimizer', 'value_optimizer']
        data = {key: value for key, value in vars(self).items() if key not in excluded}
        save_data(data, self.dir_path)

        # save model initial parameters
        save_model(self.policy, self.dir_path, 'policy_n-it{}'.format(0))
        if self.learn_value:
            save_model(self.value, self.dir_path, 'value_n-it{}'.format(0))

        # create object to store the is statistics of the learning
        is_stats = ISStatistics(
            eval_freq=1,
            eval_batch_size=self.batch_size,
            n_iterations=self.n_grad_iterations,
            iter_str='grad. it.:',
            policy_type='stoch',
            track_loss=True,
            track_ct=True,
        )
        keys_chosen = [
            'max_lengths', 'total_lengths', 'mean_fhts', 'var_fhts',
            'mean_returns', 'var_returns',
            'losses', 'loss_vars',
            'cts',
        ]

        if live_plot_freq and self.env.unwrapped.d == 1:
            mean, sigma = evaluate_stoch_policy_model(self.env.unwrapped, self.policy)
            lines = initialize_gaussian_policy_1d_figure(self.env.unwrapped, mean, sigma, policy_opt=policy_opt)

        for i in np.arange(self.n_grad_iterations+1):

            # start timer
            ct_initial = time.time()

            # sample loss function
            if self.expectation_type == 'random-time':
                loss, loss_var = self.sample_loss_random_time()
            else: # self.expectation_type == 'on-policy':
                loss, loss_var = self.sample_loss_on_policy()
            if self.learn_value:
                value_loss = self.sample_value_loss()

            # end timer
            ct_final = time.time()

            # save and log epoch 
            self.env.env.statistics_to_numpy()
            is_stats.save_epoch(i, self.env, loss=loss.detach().numpy(),
                                loss_var=loss_var, ct=ct_final - ct_initial)
            is_stats.log_epoch(i) if i % log_freq == 0 else None

            # backup models
            if backup_freq and (i + 1) % backup_freq== 0:
                save_model(self.policy, self.dir_path, 'policy_n-it{}'.format(i + 1))

            # backup statistics
            if (i + 1) % 100 == 0:
                stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
                save_data(data | stats_dict, self.dir_path)

            # update plots
            if live_plot_freq and self.env.unwrapped.d == 1 and i % live_plot_freq == 0:
                mean, sigma = evaluate_stoch_policy_model(self.env.unwrapped, self.policy)
                update_gaussian_policy_1d_figure(self.env.unwrapped, mean, sigma, lines)

        stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
        data = data | stats_dict
        save_data(data, self.dir_path)
        return True, data

    def load_backup_model(self, data, i=0):
        try:
            load_model(self.policy, data['dir_path'], file_name='policy_n-it{}'.format(i))
            return True
        except FileNotFoundError as e:
            print('There is no backup for grad. iteration {:d}'.format(i))
            return False

    def get_means_and_stds(self, data, iterations):
        env = self.env.unwrapped
        n_iterations = len(iterations)
        means = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        stds = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        for i, it in enumerate(iterations):
            self.load_backup_model(data, it)
            mean, std = evaluate_stoch_policy_model(env, self.policy)
            means[i] = mean.reshape(env.n_states, env.d)
            stds[i] = std.reshape(env.n_states, env.d)
        return means, stds
