import numpy as np
import scipy.stats as stats
import torch

from sde.langevin_sde import LangevinSDE
from hjb.hjb_solver import SolverHJB

class DoubleWellStoppingTime1D():

    def __init__(self, beta=1., alpha=1., dt=0.005, is_state_init_sampled=False):

        # environment log name
        self.name = 'doublewell-1d-st__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

        # double well potential
        self.d = 1
        self.alpha = alpha

        # sde
        self.beta = beta
        self.sigma = np.sqrt(2. / self.beta)
        self.sigma_tensor = torch.tensor(self.sigma, dtype=torch.float32)

        # Euler-Maruyama
        self.dt = dt
        self.dt_tensor = torch.tensor(self.dt, dtype=torch.float32)

        # target set
        self.lb, self.rb = 1., 2.

        # initial state
        self.is_state_init_sampled = is_state_init_sampled
        self.state_init = - np.ones((1, self.d), dtype=np.float32)

        # observation space bounds
        self.state_space_dim = 1
        self.state_space_low = -2.
        self.state_space_high = 2.

        # action space bounds
        self.action_space_dim = 1
        self.action_space_low = 0.
        self.action_space_high = 3.

    def potential(self, state):
        return self.alpha * (state**2 - 1) ** 2

    def gradient(self, state):
        return 4 * self.alpha * state * (state**2 - 1)

    def is_done(self, state):
        return np.where(state[:, 0] >= self.lb, True, False)

    def is_done_torch(self, state):
        return torch.where(state[:, 0] >= self.lb, True, False)

    def f(self, state):
        return np.ones(state.shape[0])

    def g(self, state):
        return np.zeros(state.shape[0])

    def f_torch(self, state):
        return torch.ones(state.shape[0])

    def g_torch(self, state):
        return torch.zeros(state.shape[0])

    def reset(self, batch_size=1):
        if not self.is_state_init_sampled:
            return np.full((batch_size, self.d), self.state_init)
        else:
            return np.random.uniform(self.state_space_low, self.lb, (batch_size, self.d))

    def sample_state(self, batch_size=1):
            return np.random.uniform(self.state_space_low, self.state_space_high, (batch_size, self.d))

    def sample_action(self, batch_size=1):
            return np.random.uniform(self.action_space_low, self.action_space_high, (batch_size, self.d))

    def state_action_transition_function(self, next_state, state, action, h):
        mu = state + (- self.gradient(state) + self.sigma * action) * self.dt
        std_dev = self.sigma * np.sqrt(self.dt)
        prob = stats.norm.cdf(next_state + h, mu, std_dev) \
             - stats.norm.cdf(next_state - h, mu, std_dev)
        return prob

    def reward_signal_state_action(self, state, action):
        done = self.is_done(state)
        reward = np.where(
            done,
            - self.g(state),
            - (self.f(state) + 0.5 * (np.linalg.norm(action, axis=1)**2)) * self.dt,
        )
        return reward, done

    def reward_signal_state_action_next_state(self, state, action, next_state):
        done = self.is_done(next_state)
        reward = np.where(
            done,
            - (self.f(state) + 0.5 * (np.linalg.norm(action, axis=1)**2)) * self.dt \
            - self.g(next_state),
            - (self.f(state) + 0.5 * (np.linalg.norm(action, axis=1)**2)) * self.dt,
        )
        return reward, done

    def reward_signal_state_action_torch(self, state, action):
        done = self.is_done(state)
        reward = np.where(
            done,
            - self.g(state),
            - self.f(state) * self.dt - 0.5 * (torch.linalg.norm(action, axis=1)**2) * self.dt,
        )
        return reward, done

    def reward_signal_state_action_next_state_torch(self, state, action, next_state):
        done = self.is_done_torch(next_state)
        reward = torch.where(
            done,
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor \
            - self.g_torch(next_state),
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor,
        )
        return reward, done


    def step(self, state, action):

        # batch_size
        batch_size = state.shape[0]

        # brownian increment
        dbt = np.array(np.sqrt(self.dt) * np.random.randn(batch_size, self.d), dtype=np.float32)

        # sde step
        next_state = state \
                   + (- self.gradient(state) + self.sigma * action) * self.dt \
                   + self.sigma * dbt

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        r, done = self.reward_signal_state_action_next_state(state, action, next_state)

        return next_state, r, done, dbt

    def step_vectorized_stopped(self, states, actions, idx):

        # number of trajectories which have not arrived yet
        n_not_in_ts = idx.shape[0]

        # brownian increment
        dbt = np.array(np.sqrt(self.dt) * np.random.randn(n_not_in_ts, 1), dtype=np.float32)

        # sde step
        next_states = states.copy()
        next_states[idx] = states[idx] \
                         + (- self.gradient(states[idx]) + self.sigma * actions[idx]) * self.dt \
                         + self.sigma * dbt

        # done if position x in the target set
        done = np.where(next_states >= self.lb, True, False)

        # rewards signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        batch_size = states.shape[0]
        rewards = np.zeros((batch_size, 1))
        rewards[idx] = np.where(
            done[idx],
            - 0.5 * np.power(actions[idx], 2) * self.dt - self.f(states[idx]) * self.dt -
            self.g(next_states[idx]),
            - 0.5 * np.power(actions[idx], 2) * self.dt - self.f(states[idx]) * self.dt,
        )

        return next_states, rewards, done

    def step_torch(self, state, action):

        # batch_size
        batch_size = state.shape[0]

        # brownian increment
        dt = self.dt_tensor
        dbt = torch.sqrt(dt) * torch.randn((batch_size, self.d), dtype=torch.float32)

        # sde step
        sigma = self.sigma_tensor
        next_state = state \
                   + (- self.gradient(state) + sigma * action) * dt \
                   + sigma * dbt

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        r, done = self.reward_signal_state_action_next_state_torch(state, action, next_state)

        return next_state, r, done, dbt

    def get_idx_new_in_ts(self, is_in_target_set, been_in_target_set):

        idx = np.where(
                (is_in_target_set == True) &
                (been_in_target_set == False)
        )[0]

        been_in_target_set[idx] = True

        return idx

    def get_idx_new_in_ts_torch(self, is_in_target_set, been_in_target_set):

        idx = torch.where(
                (is_in_target_set == True) &
                (been_in_target_set == False)
        )[0]

        been_in_target_set[idx] = True

        return idx

    def discretize_state_space(self, h_state):

        # discretize state space
        self.state_space_h = np.around(
            np.arange(
                self.state_space_low,
                self.state_space_high + h_state,
                h_state,
            ),
            decimals=3,
        )
        self.n_states = self.state_space_h.shape[0]
        self.h_state = h_state

        # get initial state index 
        self.get_idx_state_init()

        # get target set indices
        self.get_idx_target_set()

    def discretize_action_space(self, h_action):

        # discretize action space
        self.action_space_h = np.arange(
            self.action_space_low,
            self.action_space_high + h_action,
            h_action,
        )
        self.n_actions = self.action_space_h.shape[0]
        self.h_action = h_action

        # get null action index
        self.get_idx_null_action()


    def get_state_idx(self, state):
        return np.argmin(np.abs(self.state_space_h - state), axis=1)

    def get_state_idx_clip(self, state):
        return np.floor((
            np.clip(state, self.state_space_low, self.state_space_high - 2 * self.h_state) + self.state_space_high
        ) / self.h_state).astype(int)

    def get_action_idx(self, action):
        return np.argmin(np.abs(self.action_space_h - action), axis=1)

    def get_idx_state_init(self):
        self.idx_state_init = self.get_state_idx(self.state_init)

    def get_idx_target_set(self):
        self.idx_lb = self.get_state_idx(np.array([[self.lb]]))[0]
        self.idx_rb = self.get_state_idx(np.array([[self.rb]]))[0]
        self.idx_ts = np.where(self.state_space_h >= self.lb)[0]
        self.idx_not_ts = np.where(self.state_space_h < self.lb)[0]

    def get_idx_null_action(self):
        self.idx_null_action = self.get_action_idx(np.zeros((1, 1)))

    def get_greedy_actions(self, q_table):

        # compute greedy action by following the q-table
        idx_actions = np.argmax(q_table, axis=1)
        greedy_actions = self.action_space_h[idx_actions]

        # set actions in the target set to 0
        greedy_actions[self.idx_ts] = 0.
        return greedy_actions

    def get_hjb_solver(self, h_hjb=0.001):

        # initialize Langevin sde
        sde = LangevinSDE(
            problem_name='langevin_stop-t',
            potential_name='nd_2well',
            d=1,
            alpha=self.alpha * np.ones(1),
            beta=self.beta,
            domain=np.full((1, 2), [-2, 2]),
        )

        # load  hjb solver
        sol_hjb = SolverHJB(sde, h=h_hjb)
        sol_hjb.load()

        # if hjb solver has different discretization step coarse solution
        if sol_hjb.sde.h < self.h_state:

            # discretization step ratio
            k = int(self.h_state / sol_hjb.sde.h)
            assert self.state_space_h.shape == sol_hjb.u_opt[::k, 0].shape, ''

            sol_hjb.value_function = sol_hjb.value_function[::k]
            sol_hjb.u_opt = sol_hjb.u_opt[::k]

        return sol_hjb
