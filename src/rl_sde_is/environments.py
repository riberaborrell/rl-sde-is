import numpy as np
import scipy.stats as stats

class DoubleWellStoppingTime1D():

    def __init__(self):

        # sde
        self.beta = 1
        self.sigma = np.sqrt(2 / self.beta)

        # Euler-Maruyama
        self.dt = 0.01

        # target set
        self.lb, self.rb = 1, 2

        # initial state
        self.state_init = - np.array([1.], dtype=np.float32)

        # observation space bounds
        self.obs_space_low = -2.
        self.obs_space_high = 2.

        # action space bounds
        self.act_space_low = -5.
        self.act_space_high = 5.

    def gradient(self, state):
        return 4 * state * (state**2 - 1)

    def f(self, state):
        return 1

    def g(self, state):
        return 0

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

    def discretize_state_space(self, h_state):
        self.state_space_h = np.around(
            np.arange(
                self.obs_space_low,
                self.obs_space_high + h_state,
                h_state,
            ),
            decimals=3,
        )
        self.n_states = self.state_space_h.shape[0]
        self.h_state = h_state

    def discretize_action_space(self, h_action):
        self.action_space_h = np.arange(
            self.act_space_low,
            self.act_space_high + h_action,
            h_action,
        )
        self.n_actions = self.action_space_h.shape[0]
        self.h_action = h_action

    def get_state_idx(self, state):
        return np.argmin(np.abs(self.state_space_h - state))

    def get_action_idx(self, action):
        return np.argmin(np.abs(self.action_space_h - action))

    def get_idx_target_set(self):
        self.idx_lb = self.get_state_idx(self.lb)
        self.idx_rb = self.get_state_idx(self.rb)
