import numpy as np
import torch

from rl_sde_is.utils_numeric import *

class SdeIsEnv(object):
    '''
    '''

    def __init__(self, d: int=1, dt: float=0.005, state_space_bounds=None,
                 action_space_bounds=None, state_init_dist=None, reward_type='state-action'):

        # sde dimension
        self.d = d

        # observation space
        self.state_space_dim = self.d
        self.state_space_bounds = state_space_bounds

        # action space
        self.action_space_dim = self.d
        self.action_space_bounds = action_space_bounds

        # Euler-Maruyama
        self.dt = dt
        self.dt_tensor = torch.tensor(self.dt, dtype=tf32)

        # problem types flags
        self.is_mgf = False
        self.is_committor = False
        self.overdamped_langevin = False

        # reward type
        self.reward_type = reward_type

        # initial state distribution
        self.state_init_dist = state_init_dist

    def set_mgf_setting(self, lam=1.):
        ''' Set moment generating function of the first hitting time setting
        '''
        # set mgf problem flag
        self.is_mgf = True

        # running cost
        self.lam = lam
        self.f = lambda x: lam * np.ones(x.shape[0])
        self.f_torch = lambda x: lam * torch.ones(x.shape[0])

        # final cost
        self.g = lambda x: np.zeros(x.shape[0])
        self.g_torch = lambda x: torch.zeros(x.shape[0])

        # is done function
        self.is_done = lambda x: np.where(self.is_in_target_set(x), True, False)#.squeeze(axis=1)
        self.is_done_torch = lambda x: torch.where(self.is_in_target_set(x), True, False)#.squeeze(axis=1)

        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_mgf

    def set_committor_setting(self, epsilon=1e-10):
        ''' Set committor probability setting
        '''
        # set mgf problem flag
        self.is_committor = True

        # running cost
        self.epsilon = epsilon
        self.f = lambda x: np.zeros(x.shape[0])
        self.f_torch = lambda x: torch.zeros(x.shape[0])

        # final cost
        self.g = lambda x: np.where(
            self.is_in_target_set_b(x),
            -np.log(1+epsilon),
            -np.log(epsilon),
        ).squeeze()
        self.g_torch = lambda x: torch.where(
            self.is_in_target_set_b(x),
            -np.log(1+self.epsilon),
            -np.log(self.epsilon),
        ).squeeze()

        # is done function
        self.is_done = lambda x: np.where(
            (self.is_in_target_set_a(x) == True) | (self.is_in_target_set_b(x) == True),
            True,
            False,
        ).squeeze()

        self.is_done_torch = lambda x: torch.where(
            (self.is_in_target_set_a(x) == True) | (self.is_in_target_set_b(x) == True),
            True,
            False,
        ).squeeze()


        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_committor

    def reward_signal_state_action(self, state, action, done):
        reward = np.where(
            done,
            - self.g(state),
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt,
        )
        return reward

    def reward_signal_state_action_next_state(self, state, action, next_state, done):
        running_cost = (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt
        reward = np.where(
            done,
            - running_cost - self.g(next_state),
            - running_cost,
        )
        return reward

    def reward_signal_state_action_torch(self, state, action, done):
        reward = torch.where(
            done,
            - self.g_torch(state),
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor,
        )
        return reward

    def reward_signal_state_action_next_state_torch(self, state, action, next_state, done):
        running_cost = (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor
        reward = torch.where(
            done,
            - running_cost - self.g_torch(next_state),
            - running_cost,
        )
        return reward

    def step(self, state, action):

        # batch_size
        batch_size = state.shape[0]

        # brownian increment
        dbt = np.array(np.sqrt(self.dt) * np.random.randn(batch_size, self.d), dtype=nf32)

        # sde step
        next_state = state \
                   + (- self.gradient_fn(state) + self.sigma * action) * self.dt \
                   + self.sigma * dbt

        # reward signal r_n = r(s_n, a_n)
        if self.reward_type == 'state-action':
            done = self.is_done(state)
            r = self.reward_signal_state_action(state, action, done)

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        elif self.reward_type == 'state-action-next-state':
            done = self.is_done(next_state)
            r = self.reward_signal_state_action_next_state(state, action, next_state, done)

        elif self.reward_type == 'baseline':
            done = self.is_done(next_state)
            r = self.reward_signal_state_action_next_state_baseline(state, action, next_state, done)

        return next_state, r, done, dbt


    def step_torch(self, state, action):

        # batch_size
        batch_size = state.shape[0]

        # brownian increment
        dt = self.dt_tensor
        dbt = torch.sqrt(dt) * torch.randn((batch_size, self.d), dtype=tf32)

        # sde step
        sigma = self.sigma_tensor
        next_state = state \
                   + (- self.gradient_fn(state) + sigma * action) * dt \
                   + sigma * dbt

        # reward signal r_n = r(s_n, a_n)
        if self.reward_type == 'state-action':
            done = self.is_done_torch(state)
            r = self.reward_signal_state_action_torch(state, action, done)

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        elif self.reward_type == 'state-action-next-state':
            done = self.is_done_torch(next_state)
            r = self.reward_signal_state_action_next_state_torch(state, action, next_state, done)

        elif self.reward_type == 'baseline':
            done = self.is_done_torch(next_state)
            r = self.reward_signal_state_action_next_state_baseline_torch(state, action, next_state, done)

        return next_state, r, done, dbt

    def get_new_in_ts_idx(self, is_in_target_set, been_in_target_set):

        idx = np.where(
                (is_in_target_set == True) &
                (been_in_target_set == False)
        )[0]

        been_in_target_set[idx] = True

        return idx

    def get_new_in_ts_idx_torch(self, is_in_target_set, been_in_target_set):

        idx = torch.where(
                (is_in_target_set == True) &
                (been_in_target_set == False)
        )[0]

        been_in_target_set[idx] = True

        return idx

