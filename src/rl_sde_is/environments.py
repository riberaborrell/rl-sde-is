import numpy as np
import scipy.stats as stats
import torch

from sde.langevin_sde import LangevinSDE
from hjb.hjb_solver import SolverHJB

class DoubleWellStoppingTime1D():

    def __init__(self, is_state_init_sampled=False):

        # sde
        self.beta = 1
        self.sigma = np.sqrt(2 / self.beta)
        self.sigma_tensor = torch.tensor(self.sigma, dtype=torch.float32)

        # Euler-Maruyama
        self.dt = 0.005
        self.dt_tensor = torch.tensor(self.dt, dtype=torch.float32)

        # target set
        self.lb, self.rb = 1, 2

        # initial state
        self.is_state_init_sampled = is_state_init_sampled
        self.state_init = np.array([-1.], dtype=np.float32)

        # observation space bounds
        self.state_space_dim = 1
        self.state_space_low = -2.
        self.state_space_high = 2.

        # action space bounds
        self.action_space_dim = 1
        self.action_space_low = 0.
        self.action_space_high = 3.

    def gradient(self, state):
        return 4 * state * (state**2 - 1)

    def f(self, state):
        return 1

    def g(self, state):
        return 0

    def f_torch(self, states):
        return torch.ones(states.shape[0])

    def g_torch(self, states):
        return torch.zeros(states.shape[0])

    def state_action_transition_function(self, next_state, state, action, h):
        mu = state + (- self.gradient(state) + self.sigma * action) * self.dt
        std_dev = self.sigma * np.sqrt(self.dt)
        prob = stats.norm.cdf(next_state + h, mu, std_dev) \
             - stats.norm.cdf(next_state - h, mu, std_dev)
        return prob

    def reward_signal(self, state, action):
        lb, rb = 1, 2
        dt = 0.01

        reward = np.where(
            state < lb,
            - dt - 0.5 * (action**2) * dt,
            0,
        )
        return reward

    def reset(self):
        if not self.is_state_init_sampled:
            return self.state_init.copy()
        else:
            return np.random.uniform(self.state_space_low, self.lb, (1,))

    def reset_vectorized(self, batch_size):
        if not self.is_state_init_sampled:
            return np.full((batch_size, 1), self.state_init)
        else:
            return np.random.uniform(self.state_space_low, self.lb, (batch_size, 1))

    def step(self, state, action):

        # brownian increment
        dbt = np.array(np.sqrt(self.dt) * np.random.randn(1), dtype=np.float32)

        # sde step
        next_state = state \
                   + (- self.gradient(state) + self.sigma * action) * self.dt \
                   + self.sigma * dbt

        # done if position x in the target set
        done = bool(next_state > self.lb and next_state < self.rb)

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        r = np.where(
            done,
            - 0.5 * np.power(action, 2)[0] * self.dt - self.f(state) * self.dt - self.g(next_state),
            - 0.5 * np.power(action, 2)[0] * self.dt - self.f(state) * self.dt,
        )

        return next_state, r, done

    def step_torch(self, state, action):

        # brownian increment
        dt = self.dt_tensor
        dbt = torch.sqrt(dt) * torch.randn(1, dtype=torch.float32)

        # sde step
        sigma = self.sigma_tensor
        next_state = state \
                   + (- self.gradient(state) + sigma * action) * dt \
                   + sigma * dbt

        # done if position x in the target set
        done = torch.where(next_state > self.lb, True, False)

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        r = torch.where(
            done,
            - 0.5 * torch.pow(action, 2)[0] * dt - self.f(state) * dt - self.g(next_state),
            - 0.5 * torch.pow(action, 2)[0] * dt - self.f(state) * dt,
        )

        return next_state, r, done, dbt

    def step_vectorized(self, states, actions):

        # batch_size
        batch_size = states.shape[0]

        # brownian increment
        dbt = np.array(np.sqrt(self.dt) * np.random.randn(batch_size, 1), dtype=np.float32)

        # sde step
        next_states = states \
                    + (- self.gradient(states) + self.sigma * actions) * self.dt \
                    + self.sigma * dbt

        # done if position x in the target set
        done = np.where(next_states > self.lb, True, False)

        # rewards signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        rewards = np.where(
            done,
            - 0.5 * np.power(actions, 2) * self.dt - self.f(states) * self.dt - self.g(next_states),
            - 0.5 * np.power(actions, 2) * self.dt - self.f(states) * self.dt,
        )

        return next_states, rewards, done

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
        done = np.where(next_states > self.lb, True, False)

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

    def step_vectorized_torch(self, states, actions):

        # batch_size
        batch_size = states.shape[0]

        # brownian increment
        dt = self.dt_tensor
        dbt = torch.sqrt(dt) * torch.randn((batch_size, 1), dtype=torch.float32)

        # sde step
        sigma = self.sigma_tensor
        next_states = states \
                    + (- self.gradient(states) + sigma * actions) * dt \
                    + sigma * dbt

        # done if position x in the target set
        done = torch.where(next_states > self.lb, True, False)

        # rewards signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        rewards = torch.where(
            done,
            - 0.5 * torch.pow(actions, 2) * dt - self.f(states) * dt - self.g(next_states),
            - 0.5 * torch.pow(actions, 2) * dt - self.f(states) * dt,
        )

        return next_states, rewards, done, dbt

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
        return np.argmin(np.abs(self.state_space_h - state))

    def get_states_idx_vectorized(self, states):
        return np.argmin(np.abs(self.state_space_h - states), axis=1)

    def get_state_idx_clip(self, state):
        return np.floor((
            np.clip(state, self.state_space_low, self.state_space_high - 2 * self.h_state) + self.state_space_high
        ) / self.h_state).astype(int)

    def get_action_idx(self, action):
        return np.argmin(np.abs(self.action_space_h - action))

    def get_idx_state_init(self):
        self.idx_state_init = self.get_state_idx(self.state_init)

    def get_idx_target_set(self):
        self.idx_lb = self.get_state_idx(self.lb)
        self.idx_rb = self.get_state_idx(self.rb)

    def get_idx_null_action(self):
        self.idx_null_action = self.get_action_idx(0.)

    def get_greedy_actions(self, q_table):

        # compute greedy action by following the q-table
        idx_actions = np.argmax(q_table, axis=1)
        greedy_actions = self.action_space_h[idx_actions]

        # set actions in the target set to 0
        greedy_actions[self.idx_lb:] = 0.
        return greedy_actions

    def get_hjb_solver(self, h_hjb=0.01):

        # initialize Langevin sde
        sde = LangevinSDE(
            problem_name='langevin_stop-t',
            potential_name='nd_2well',
            d=1,
            alpha=np.ones(1),
            beta=1.,
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
