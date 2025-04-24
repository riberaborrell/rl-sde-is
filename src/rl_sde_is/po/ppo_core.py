# script adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym_sde_is.wrappers.record_episode_statistics import RecordEpisodeStatisticsVect
from gym_sde_is.wrappers.save_episode_trajectory import SaveEpisodeTrajectoryVect
from gym_sde_is.utils.butane import compute_state_vect, compute_dihedral_vect

from rl_sde_is.po.models import ActorCriticModel
from rl_sde_is.utils.approximate_methods import evaluate_stoch_policy_model, evaluate_value_function_model
from rl_sde_is.utils.is_statistics import ISStatistics
from rl_sde_is.utils.numeric import cumsum_numpy as cumsum, normalize_array
from rl_sde_is.utils.path import load_data, save_data, save_model, load_model, get_ppo_dir_path
from rl_sde_is.utils.plots import initialize_gaussian_policy_1d_figure, update_gaussian_policy_1d_figure

class PPO:
    def __init__(self, env, gamma=1.0, policy_noise_init=1.0, n_layers=2,
                 d_hidden_layer=32, optim_type='sgd', batch_size=100, lr=1e-2,
                 n_iterations=1000, n_mini_batches=32, update_epochs=10, max_grad_norm=0.5,
                 norm_adv=True, clip_vloss=True, clip_coef=0.2, ent_coef=0.,
                 vf_coef=0.5, target_kl=None, seed=None, cuda=False, torch_deterministic=True):

        # agent name
        self.agent = 'ppo'

        # environments
        self.env = env

        # discount
        self.gamma = gamma

        # cuda device
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.torch_deterministic = torch_deterministic

        # get state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # stochastic policy
        self.policy_noise_init = policy_noise_init
        self.n_layers = n_layers
        self.d_hidden_layer = d_hidden_layer

        # initialize ppo agent. actor and critic networks
        hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
        self.model = ActorCriticModel(
            self.state_dim, self.action_dim, hidden_sizes,
                activation=nn.Tanh(), std_init=policy_noise_init, seed=seed,
        ).to(self.device)

        # stochastic gradient descent
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.n_mini_batches = n_mini_batches
        self.lr = lr
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm

        # optimizer
        self.optim_type = optim_type
        if self.optim_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        elif self.optim_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError('The optimizer {optim} is not implemented')

        # normalize advantages flag
        self.norm_adv = norm_adv

        # value function
        self.clip_vloss = clip_vloss

        # clipping
        self.clip_coef = clip_coef

        # loss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        # kl divergence
        self.target_kl = target_kl

        # seed 
        self.seed = seed

    def sample_trajectories(self):

        #TODO: adapt for batch_size_z
        K = self.batch_size

        # preallocate lists to store logprobs and values 
        rewards, log_probs, values = [], [], []

        # initialization
        state, _ = self.env.reset(options={'batch_size': K})

        # terminal state flag
        done = np.full((K,), False)
        while not done.all():

            # sample action
            state_torch = torch.FloatTensor(state)
            action, log_prob = self.model.sample_action(state_torch, log_prob=True)
            with torch.no_grad():
                value = self.model.get_value(state_torch)

            # save logprobs and value
            log_probs.append(log_prob)
            values.append(value)

            # env step
            state, r, _, truncated, _ = self.env.step_vect(action)
            done = np.logical_or(self.env.unwrapped.been_terminated, truncated)

            # save rewards
            rewards.append(r)

        # convert to numpy arrays
        log_probs = np.stack(log_probs)
        values = np.stack(values).reshape(-1, K)
        rewards = np.stack(rewards)

        # compute returns and advantages
        trajs_log_probs, trajs_values, returns, advantages = [], [], [], []

        # get time steps
        time_steps = self.env.get_wrapper_attr('lengths')

        # estimate advantages by computeing the temporal difference of the value function 
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        deltas = np.vstack((deltas, np.empty((1, K))))

        for i in range(self.batch_size):
            n_final = time_steps[i]
            trajs_log_probs.append(log_probs[:n_final, i])
            trajs_values.append(values[:n_final, i])

            # compute returns
            returns.append(cumsum(rewards[:n_final, i]))

            # correct temporal difference value at the last time step
            deltas[n_final-1, i] = rewards[n_final-1, i] - values[n_final-1, i]

            # store advantages
            advantages.append(deltas[:n_final, i])

        return np.vstack(self.env.trajs_states), np.vstack(self.env.trajs_actions), np.hstack(trajs_log_probs), \
               np.hstack(trajs_values), np.hstack(returns), \
               np.hstack(advantages)

    def run_ppo(self, policy_opt=None, value_function_opt=None, backup_freq=None,
                live_plot_freq=None, log_freq=100, load=False):

        # get dir path
        self.dir_path = get_ppo_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(self.dir_path)

        # save algorithm parameters
        excluded = ['env', 'model', 'optimizer', 'device']
        data = {key: value for key, value in vars(self).items() if key not in excluded}
        save_data(data, self.dir_path)

        # vectorized environment
        self.env = RecordEpisodeStatisticsVect(self.env, self.batch_size)

        # save states, action and rewards
        self.env = SaveEpisodeTrajectoryVect(self.env, self.batch_size)

        # create object to store the is statistics of the learning
        stats = ISStatistics(
            eval_freq=1,
            eval_batch_size=self.batch_size,
            n_iterations=self.n_iterations,
            iter_str='Iter.:',
            policy_type='stoch',
            #track_pg_updates=True,
            track_ct=True,
        )
        keys_chosen = [
            'max_lengths', 'total_lengths', 'mean_fhts', 'var_fhts',
            'mean_returns', 'var_returns',
            #'n_grad_updates', 'cts',
            'cts',
        ]

        # save model initial parameters
        save_model(self.model, self.dir_path, 'model_n-it{}'.format(0))

        if live_plot_freq and self.env.unwrapped.d == 1:
            mean, sigma = evaluate_stoch_policy_model(self.env.unwrapped, self.model.actor)
            lines = initialize_gaussian_policy_1d_figure(self.env.unwrapped, mean, sigma, policy_opt=policy_opt)

        for i in range(self.n_iterations):

            # start timer
            ct_initial = time.time()

            # sample trajectories
            states, actions, log_probs, values, returns, advantages = self.sample_trajectories()

            # compute relative coordinates
            if 'butane' in self.env.unwrapped.name:
                states_rel = compute_dihedral_vect(states) if self.env.unwrapped.is_reduced else compute_state_vect(states)

            # convert to torch tensors
            states = torch.Tensor(states).to(self.device)
            actions = torch.Tensor(actions).to(self.device)
            log_probs = torch.Tensor(log_probs).to(self.device)
            values = torch.Tensor(values).to(self.device)
            returns = torch.Tensor(returns).to(self.device)
            advantages = torch.Tensor(advantages).to(self.device)
            if 'butane' in self.env.unwrapped.name:
                states_rel = torch.Tensor(states_rel)

            # n total steps in batch
            n_total_steps = states.shape[0]
            mini_batch_size = n_total_steps // self.n_mini_batches
            last_mini_batch_size = n_total_steps % self.n_mini_batches

            # Optimizing the policy and value network

            # batch indices
            b_inds = np.arange(n_total_steps, dtype=np.int32)
            clip_fracs = []

            # count policy gradient updates in the iteration
            n_pg_updates = 0

            for epoch in range(self.update_epochs):

                # shuffle data
                np.random.shuffle(b_inds)

                for j in range(0, self.n_mini_batches):

                    # get mini batch indices
                    start = j * mini_batch_size
                    end = start + mini_batch_size if j < self.n_mini_batches - 1 else n_total_steps
                    mb_inds = b_inds[start:end]

                    # compute new values of the updated critic network and
                    # new log probs of the updated gaussian distribution (actor network)
                    mb_states = states[mb_inds] if 'butane' not in self.env.unwrapped.name else states_rel[mb_inds]
                    new_values = self.model.critic(mb_states)
                    dist, new_log_probs = self.model.actor(mb_states, actions[mb_inds])

                    # compute entropy of the new distribution
                    entropy = dist.entropy().sum(1)

                    # compute the ratio of the new and old log probs
                    log_ratio = new_log_probs - log_probs[mb_inds]
                    ratio = log_ratio.exp()

                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    with torch.no_grad():
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = normalize_array(mb_advantages, eps=1e-8)

                    # policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # value loss
                    new_values = new_values.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (new_values - returns[mb_inds]) ** 2
                        v_clipped = values[mb_inds] + torch.clamp(
                            new_values - values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_values - returns[mb_inds]) ** 2).mean()

                    # entropy loss (entropy bonus for exploration)
                    entropy_loss = entropy.mean()

                    # total loss
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    n_pg_updates += 1

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            # end timer
            ct_final = time.time()

            # save and log epoch 
            self.env.env.statistics_to_numpy()
            #stats.save_epoch(i, initial_returns, time_steps, n_pg_updates=n_pg_updates, ct=ct_final - ct_initial)
            stats.save_epoch(i, self.env, ct=ct_final - ct_initial)
            stats.log_epoch(i) if i % log_freq == 0 else None

            # backup models
            if backup_freq and (i + 1) % backup_freq== 0:
                save_model(self.model, self.dir_path, 'model_n-it{}'.format(i + 1))

            # backup statistics
            if (i + 1) % 100 == 0:
                stats_dict = {key: stats.__dict__[key] for key in keys_chosen}
                save_data(data | stats_dict, self.dir_path)

            # update plots
            if live_plot_freq and self.env.unwrapped.d == 1 and i % live_plot_freq == 0:
                mean, sigma = evaluate_stoch_policy_model(self.env.unwrapped, self.model.actor)
                update_gaussian_policy_1d_figure(self.env.unwrapped, mean, sigma, lines)

        stats_dict = {key: value for key, value in vars(stats).items() if key in keys_chosen}
        data = data | stats_dict
        save_data(data, self.dir_path)

        return True, data

    def load_backup_model(self, data, i=0):
        try:
            load_model(self.model, data['dir_path'], file_name='model_n-it{}'.format(i))
            return True
        except FileNotFoundError as e:
            print('There is no backup for iteration {:d}'.format(i))
            return False

    def eval_model_state_space(self, data, iterations):
        env = self.env.unwrapped
        n_iterations = len(iterations)
        value_functions = np.empty((n_iterations, env.n_states), dtype=np.float32)
        means = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        stds = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        for i, it in enumerate(iterations):
            self.load_backup_model(data, it)
            value_functions[i] = evaluate_value_function_model(env, self.model.critic).squeeze()
            mean, std = evaluate_stoch_policy_model(env, self.model.actor)
            means[i] = mean.reshape(env.n_states, env.d)
            stds[i] = std.reshape(env.n_states, env.d)

        return value_functions, means, stds
