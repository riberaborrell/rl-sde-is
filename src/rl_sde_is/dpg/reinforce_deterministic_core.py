import functools
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
from gym_sde_is.wrappers.save_episode_trajectory import SaveEpisodeTrajectoryVect
from gym_sde_is.utils.evaluate import evaluate_policy_torch_vect
from gym_sde_is.utils.butane import * #compute_state_vect, compute_force_from_action_vect_torch

from rl_sde_is.dpg.dpg_utils import DeterministicPolicy, ValueFunction
from rl_sde_is.dpg.replay_memories import ReplayMemoryModelBasedDPG as Memory
from rl_sde_is.utils.schedulers import simple_lr_schedule
from rl_sde_is.utils.approximate_methods import evaluate_det_policy_model, \
                                                evaluate_time_dependent_det_policy_model, \
                                                evaluate_value_function_model, \
                                                train_deterministic_policy_from_hjb
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.numeric import dot_vect, cumsum_numpy as cumsum
from rl_sde_is.utils.path import get_reinforce_det_dir_path, load_data, save_data, \
                                 save_model, load_model
from rl_sde_is.utils.plots import *


class ReinforceDeterministic:
    def __init__(self, env, gamma=1.0, expectation_type='random-time', return_type='initial-return',
                 estimate_z=True, theta_init='null', n_layers=2, d_hidden_layer=32,
                 optim_type='sgd', batch_size=100, mini_batch_size_type='adaptive',
                 mini_batch_size=1, lr=1e-2, learn_value=False, lr_value=None,
                 n_grad_iterations=100, seed=None, scheduled_lr=False, lr_final=None,
                 norm_returns=True, cuda=False):


        if expectation_type == 'on-policy' and mini_batch_size is None:
            raise ValueError('The mini_batch_size must be provided when using on-policy')

        # agent
        self.agent = 'reinforce-det-{}'.format(expectation_type)

        # environment
        self.env = env

        # discount
        self.gamma = gamma

        # get state and action dimensions
        self.state_dim = env.unwrapped.d_state
        self.action_dim = env.unwrapped.d_action

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

        # deterministic policy
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

        # scheduled lr
        self.scheduled_lr=scheduled_lr
        self.lr_final=lr_final

        # cuda device
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

        # get dimensions of each layer
        d_hidden_layers = [self.d_hidden_layer for i in range(self.n_layers-1)]

        # initialize policy model 
        self.model = DeterministicPolicy(
            state_dim=self.state_dim, action_dim=self.action_dim,
             hidden_sizes=d_hidden_layers, activation=nn.Tanh(),
        ).to(self.device)

        # initialize value function model
        self.value = ValueFunction(
            state_dim=self.state_dim, hidden_sizes=d_hidden_layers, activation=nn.Tanh(),
        ).to(self.device) if self.learn_value else None

        # seed
        self.seed = seed

    def sample_trajectories(self):

        # evaluate policy. Trajectories are stored and statistics are computed
        evaluate_policy_torch_vect(self.env, self.model, self.batch_size)

        # get states, dbts and returns
        states, dbts, returns = [], [], []
        for i in range(self.batch_size):

            # states and dbts
            states.append(self.env.trajs_states[i][:-1])
            dbts.append(self.env.trajs_dbts[i][:-1])

            # compute initial returns
            if self.return_type == 'initial-return':
                returns.append(
                    np.full(self.env.get_wrapper_attr('lengths')[i]-1, self.env.get_wrapper_attr('returns')[i])
                )

            # compute n-step returns
            else: # retrun_type == 'n-return'
                returns.append(cumsum(self.env.trajs_rewards[i])[1:])

        return np.vstack(states), np.vstack(dbts), np.hstack(returns)

    def sample_loss_random_time(self):

        # sample trajectories
        states, dbts, returns = self.sample_trajectories()

        if 'butane' not in self.env.unwrapped.name:

            # compute actions following the policy
            states = torch.FloatTensor(states)
            actions = self.model.forward(states)
        else:

            # compute relative coordinates
            states_rel = compute_dihedral_vect(states) if self.env.unwrapped.is_reduced else compute_state_vect(states)

            # torchify states
            states = torch.FloatTensor(states)
            states_rel = torch.FloatTensor(states_rel)

            # compute relative actions following the model
            actions_rel = self.model.forward(states_rel)

            # compute absolute actions
            n_actions = actions_rel.shape[0]
            if self.env.unwrapped.is_reduced:
                actions = compute_force_from_dihedral_action_vect_torch(states, actions_rel).view(n_actions, -1)
            else:
                actions = compute_force_from_action_vect_torch(states, actions_rel).view(n_actions, -1)

        # torchify dbts and returns
        dbts = torch.FloatTensor(dbts)
        returns = torch.FloatTensor(returns)

        # compute girsanov deterministic and stochastic integrals
        girs_det_int = 0.5 * torch.linalg.norm(actions, axis=1).pow(2) * self.env.unwrapped.dt
        girs_stoch_int = dot_vect(dbts, actions)

        # calculate loss
        phi = girs_det_int - returns * girs_stoch_int
        loss = phi.sum() / self.batch_size
        with torch.no_grad():
            loss_var = phi.var().numpy()

        # reset gradients, compute gradients and update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # scheduler
        self.scheduler.step()

        return loss, loss_var

    def sample_loss_on_policy(self):

        # sample trajectories
        states, dbts, returns = self.sample_trajectories()

        # initialize memory
        memory = Memory(size=states.shape[0]+1, state_dim=self.state_dim, action_dim=self.action_dim)

        # store experiences in memory
        memory.store_vectorized(states, dbts, returns=returns)

        # sample batch of experiences from memory
        if self.mini_batch_size_type == 'adaptive':
            mini_batch_size = round(memory.size / self.mini_batch_size)
        else:
            mini_batch_size = self.mini_batch_size
        batch = memory.sample_batch(mini_batch_size)

        # compute actions following the policy
        actions = self.model.forward(batch['states'])

        # estimate mean trajectory length
        mean_length = self.env.get_wrapper_attr('lengths').mean() if self.estimate_z else 1

        # compute girsanov deterministic and stochastic integrals
        girs_det_int = 0.5 * torch.linalg.norm(actions, axis=1).pow(2) * self.env.unwrapped.dt
        girs_stoch_int = dot_vect(batch['dbts'], actions)

        # calculate loss
        phi = girs_det_int - batch['returns'] * girs_stoch_int
        loss = phi.mean()
        with torch.no_grad():
            loss_var = phi.var().numpy()

        # reset and compute actor gradients
        self.optimizer.zero_grad()
        loss.backward()

        # scale gradients before updating parameters
        if self.estimate_z:
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad *= mean_length

        # scale learning rate
        #optimizer.param_groups[0]['lr'] *= mean_length

        #update parameters
        self.optimizer.step()

        # re-scale learning rate back
        #optimizer.param_groups[0]['lr'] /= mean_length

        return loss, loss_var

    def sample_value_loss(self):

        # compute target value
        with torch.no_grad():

            # value function next
            next_states = np.vstack(self.env.trajs_states)[1:]
            next_states = np.vstack((next_states, np.zeros((1, self.state_dim))))
            next_states = torch.FloatTensor(next_states)
            v_next = self.value.forward(next_states)

            # compute target (using target networks)
            done = np.hstack(self.env.trajs_dones)
            done = torch.tensor(done)
            d = torch.where(done, 1., 0.)
            rewards = np.hstack(self.env.trajs_rewards)
            rewards = torch.FloatTensor(rewards)
            v_target = rewards + (1. - d) * v_next

        # compute current q-value
        states = np.vstack(self.env.trajs_states)
        states = torch.FloatTensor(states)
        v_current = self.value.forward(states)

        # compute loss
        loss = (v_current - v_target).pow(2).mean()

        # reset gradients and update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def run_agent(self, backup_freq=None, live_plot_freq=None, log_freq=100,
                  policy_opt=None, value_function_opt=None, load=False):

        # get dir path
        self.dir_path = get_reinforce_det_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(self.dir_path)

        # set seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # vectorized environment
        self.env = RecordEpisodeStatisticsVect(self.env, self.batch_size)
        track_dones = True if self.learn_value else False
        self.env = SaveEpisodeTrajectoryVect(self.env, self.batch_size, track_actions=False,
                                        track_rewards=True, track_dones=track_dones, track_dbts=True)

        # define optimizer/s
        if self.optim_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optim_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError('The optimizer {optim} is not implemented')

        # define scheduler
        if self.scheduled_lr:
            lr_schedule = functools.partial(simple_lr_schedule, lr_init=lr,
                                            lr_final=lr_final, n_iter=self.n_grad_iterations+1)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule)
        else:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda it: 1)

        if self.learn_value:
            self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr_value)

        # train params to fit hjb solution
        if self.theta_init == 'hjb':
            train_deterministic_policy_from_hjb(self.env, self.model, policy_opt, load=True)

        # save algorithm parameters
        excluded = ['env', 'policy', 'value', 'optimizer', 'value_optimizer', 'scheduler']
        data = {key: value for key, value in vars(self).items() if key not in excluded}
        save_data(data, self.dir_path)

        # save model initial parameters
        save_model(self.model, self.dir_path, 'model_n-it{}'.format(0))
        if self.learn_value:
            save_model(self.value, self.dir_path, 'value_n-it{}'.format(0))

        # create object to store the is statistics of the learning
        is_stats = ISStatistics(
            eval_freq=1,
            eval_batch_size=self.batch_size,
            n_iterations=self.n_grad_iterations,
            iter_str='grad. it.:',
            policy_type='det',
            track_loss=True,
            track_ct=True,
            track_lr=True,
        )
        keys_chosen = [
            'max_lengths', 'total_lengths', 'mean_fhts', 'var_fhts',
            'mean_returns', 'var_returns',
            'mean_I_us', 'var_I_us', 're_I_us',
            'losses', 'loss_vars',
            'cts', 'lrs',
        ]

        # initialize live figures
        if live_plot_freq:
            figs_placeholder = self.initialize_figures(policy_opt, value_function_opt)

        for i in np.arange(self.n_grad_iterations+1):

            # start timer
            ct_initial = time.time()

            # compute model based policy effective loss
            if self.expectation_type == 'random-time':
                loss, loss_var = self.sample_loss_random_time()
            else: # expectation_type == 'on-policy'
                loss, loss_var = self.sample_loss_on_policy()

            if self.learn_value:
                value_loss = self.sample_value_loss()

            # end timer
            ct_final = time.time()

            # save and log epoch 
            self.env.env.statistics_to_numpy()
            is_stats.save_epoch(i, self.env, loss=loss.detach().numpy(),
                                loss_var=loss_var, ct=ct_final - ct_initial,
                                lr=self.scheduler.get_last_lr()[0])
            is_stats.log_epoch(i) if i % log_freq == 0 else None

            # backup models
            if backup_freq is not None and (i + 1) % backup_freq == 0:
                save_model(self.model, self.dir_path, 'model_n-it{}'.format(i + 1))
                if self.learn_value:
                    save_model(self.value, self.dir_path, 'value_n-it{}'.format(i + 1))

            # backup statistics
            if (i + 1) % 100 == 0:
                stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
                save_data(data | stats_dict, self.dir_path)

            # update figure
            if live_plot_freq and i % live_plot_freq == 0:
                self.update_figures(figs_placeholder)

        # add learning results
        stats_dict = {key: is_stats.__dict__[key] for key in keys_chosen}
        data = data | stats_dict
        save_data(data, self.dir_path)
        return True, data

    def load_backup_model(self, data, i=0):
        try:
            load_model(self.model, data['dir_path'], file_name='model_n-it{}'.format(i))
            if data['learn_value']:
                load_model(self.value, data['dir_path'], file_name='value_n-it{}'.format(i))
            return True
        except FileNotFoundError as e:
            print('There is no backup for grad. iteration {:d}'.format(i))
            return False

    def get_policies(self, data, iterations):
        env = self.env.unwrapped
        n_iterations = len(iterations)
        policies = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        for i, it in enumerate(iterations):
            self.load_backup_model(data, it)
            policies[i] = evaluate_det_policy_model(env, data['model'])
        return policies

    def get_time_dependent_policies(self, data, iterations, time_step):
        env = self.env.unwrapped
        n_iterations = len(iterations)
        policies = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        for i, it in enumerate(iterations):
            self.load_backup_model(data, it)
            policies[i] = evaluate_time_dependent_det_policy_model(env, data['model'], time_step)
        return policies

    def get_value_functions(self, data, iterations):
        env = self.env.unwrapped
        n_iterations = len(iterations)
        value_functions = np.empty((n_iterations, env.n_states), dtype=np.float32)
        for i, it in enumerate(iterations):
            self.load_backup_model(data, it)
            value_functions[i] = evaluate_value_function_model(env, data['value'])
        return value_functions

    def initialize_figures(self, policy_opt, value_function_opt):
        # evaluate policy and value function
        env = self.env.unwrapped
        policy = evaluate_det_policy_model(env, self.model).reshape(env.state_space_h.shape)
        if self.value is not None:
            values = evaluate_value_function_model(env, self.value).reshape(env.state_space_h.shape[:-1])

        if env.d == 1:
            policy_line = initialize_det_policy_1d_figure(env, policy, policy_opt=policy_opt)
            value_line = initialize_value_function_1d_figure(env, values, value_function_opt) \
                         if self.value is not None else None
            return policy_line, value_line
        elif env.d == 2:
            policy_quiver = initialize_det_policy_2d_figure(env, policy, policy_opt)
            value_im = initialize_value_function_2d_figure(env, values) \
                       if value is not None else None
            return policy_quiver, value_im


    def update_figures(self, figs_placeholder):

        # evaluate policy and value function
        env = self.env.unwrapped
        policy = evaluate_det_policy_model(env, self.model).reshape(env.state_space_h.shape)
        if self.value is not None:
            values = evaluate_value_function_model(env, self.value).reshape(env.state_space_h.shape[:-1])

        if env.d == 1:
            policy_line, value_line = figs_placeholder
            update_det_policy_1d_figure(env, policy, policy_line)
            if self.value is not None:
                update_value_function_1d_figure(env, values, value_line)

        elif env.d == 2:
            policy_quiver, value_im = figs_placeholder
            update_det_policy_2d_figure(env, policy, policy_quiver)
            if self.value is not None:
                update_value_function_2d_figure(env, values, value_im)
