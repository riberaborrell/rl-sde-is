import numpy as np
import scipy.stats as stats
import torch

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
        return np.where((state[:, 0] >= self.lb) & (state[:, 0] <= self.rb), True, False)

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
            return np.full((batch_size, self.d), np.random.uniform(self.state_space_low, self.lb, (self.d,)))
            #return np.full((batch_size, self.d), np.random.uniform(self.state_space_low, self.state_space_high, (self.d,)))

    def reset_done(self, states, done):
        if done.any():
            idx = np.where(done)[0]
            states[idx] = self.reset(batch_size=done.sum())

    def sample_state(self, batch_size=1):
            return np.random.uniform(self.state_space_low, self.state_space_high, (batch_size, self.d))

    def sample_state_ts(self, batch_size=1):
            return np.random.uniform(self.lb, self.state_space_high, (batch_size, self.d))

    def sample_action(self, batch_size=1):
            return np.random.uniform(self.action_space_low, self.action_space_high, (batch_size, self.d))

    def state_action_transition_function(self, next_states, state, action, h):

        # compute mean and standard deviation
        mu = state + (- self.gradient(state) + self.sigma * action) * self.dt
        std_dev = self.sigma * np.sqrt(self.dt)

        # compute probabilities
        prob = stats.norm.cdf(next_states + h, mu, std_dev) \
             - stats.norm.cdf(next_states - h, mu, std_dev)

        # add tail probabilities
        prob_left_tail = stats.norm.cdf(next_states[0] - h, mu, std_dev)
        prob_right_tail = 1 - stats.norm.cdf(next_states[-1] + h, mu, std_dev)
        prob[0] += prob_left_tail
        prob[-1] += prob_right_tail
        return prob

    def reward_signal_state_action(self, state, action, done):
        reward = np.where(
            done,
            - self.g(state),
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt,
        )
        return reward

    def reward_signal_state_action_next_state(self, state, action, next_state, done):
        reward = np.where(
            done,
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt \
            - self.g(next_state),
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt,
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
        reward = torch.where(
            done,
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor \
            - self.g_torch(next_state),
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor,
        )
        return reward


    def step(self, state, action, reward_type='state-action'):

        # batch_size
        batch_size = state.shape[0]

        # brownian increment
        dbt = np.array(np.sqrt(self.dt) * np.random.randn(batch_size, self.d), dtype=np.float32)

        # sde step
        next_state = state \
                   + (- self.gradient(state) + self.sigma * action) * self.dt \
                   + self.sigma * dbt

        # reward signal r_n = r(s_n, a_n)
        if reward_type == 'state-action':
            done = self.is_done(state)
            r = self.reward_signal_state_action(state, action, done)

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        elif reward_type == 'state-action-next-state':
            done = self.is_done(next_state)
            r = self.reward_signal_state_action_next_state(state, action, next_state, done)

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

        batch_size = states.shape[0]
        rewards = np.zeros((batch_size, 1))

        # rewards signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        rewards[idx] = np.where(
            done[idx],
            - (self.f(states[idx]) + 0.5 * np.power(actions[idx], 2)) * self.dt \
            - self.g(next_states[idx]),
            - (self.f(states[idx]) + 0.5 * np.power(actions[idx], 2)) * self.dt,
        )

        # rewards signal r_n = r(s_n, a_n)
        #rewards[idx] = np.where(
        #    done[idx],
        #    - self.g(states[idx]),
        #    - (self.f(states[idx]) + 0.5 * np.power(actions[idx], 2)) * self.dt,
        #)

        return next_states, rewards, done, dbt

    def step_torch(self, state, action, reward_type='state-action'):

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

        # reward signal r_n = r(s_n, a_n)
        if reward_type == 'state-action':
            done = self.is_done_torch(state)
            r = self.reward_signal_state_action_torch(state, action, done)

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        elif reward_type == 'state-action-next-state':
            done = self.is_done_torch(next_state)
            r = self.reward_signal_state_action_next_state_torch(state, action, next_state, done)

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

    def set_action_space_bounds(self):

        if self.alpha == 1. and self.beta == 1:
            a = 3
        elif self.alpha == 5. and self.beta == 1:
            a = 8
        elif self.alpha == 1. and self.beta == 4:
            a = 5
        elif self.alpha == 10. and self.beta == 1:
            a = 20
        else:
            return

        self.action_space_low = - a
        self.action_space_high = a

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
        self.get_state_init_idx()

        # get target set indices
        self.get_target_set_idx()

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
        self.get_null_action_idx()

    def get_state_idx(self, state):
        ''' get index of the corresponding discretized state
        '''

        # array convertion
        state = np.asarray(state)

        # scalar input
        if state.ndim == 0:
            state = state[np.newaxis, np.newaxis]

        # array input
        elif state.ndim == 1:
            state = state[np.newaxis]

        return self.get_state_idx_truncate(state)
        #return self.get_state_idx_min(state)

    def get_state_idx_truncate(self, state):
        state = np.clip(state, self.state_space_low, self.state_space_high)
        idx = np.floor((state - self.state_space_low) / self.h_state).astype(int)
        return idx[:, 0]

    def get_state_idx_min(self, state):
        return np.argmin(np.abs(self.state_space_h - state), axis=1)

    def get_action_idx(self, action):
        ''' get index of the corresponding discretized action
        '''

        # array convertion
        action = np.asarray(action)

        # scalar input
        if action.ndim == 0:
            action = action[np.newaxis, np.newaxis]

        # array input
        elif action.ndim == 1:
            action = action[np.newaxis]

        return self.get_action_idx_truncate(action)
        #return self.get_action_idx_min(action)

    def get_action_idx_truncate(self, action):
        action = np.clip(action, self.action_space_low, self.action_space_high)
        idx = np.floor((action - self.action_space_low) / self.h_action).astype(int)
        return idx[:, 0]

    def get_action_idx_min(self, action):
        return np.argmin(np.abs(self.action_space_h - action), axis=1)

    def get_state_init_idx(self):
        self.state_init_idx = self.get_state_idx(self.state_init)

    def get_target_set_idx(self):
        self.is_in_ts = (self.state_space_h >= self.lb) & (self.state_space_h <= self.rb)
        self.lb_idx = self.get_state_idx(np.array([[self.lb]]))[0]
        self.rb_idx = self.get_state_idx(np.array([[self.rb]]))[0]
        self.ts_idx = np.where(self.is_in_ts)[0]
        self.not_ts_idx = np.where(np.invert(self.is_in_ts))[0]

    def get_null_action_idx(self):
        self.null_action_idx = self.get_action_idx(np.zeros((1, self.d)))

    def get_greedy_actions(self, q_table):

        # compute greedy action by following the q-table
        actions_idx = np.argmax(q_table, axis=1)
        greedy_actions = self.action_space_h[actions_idx]

        # set actions in the target set to 0
        greedy_actions[self.ts_idx] = 0.
        return greedy_actions

    def discretize_state_action_space(self):

        # slice according to the state space and action space bounds and discretization step
        slice_i_state = slice(self.state_space_low, self.state_space_high + self.h_state, self.h_state)
        slice_i_action = slice(self.action_space_low, self.action_space_high + self.h_action, self.h_action)

        # number of state-actions pairs in the grid
        self.n_states_actions = self.n_states * self.n_actions

        # get state action space grid
        m_grid = np.mgrid[[
            slice_i_state,
            slice_i_action,
        ]]
        self.state_action_space_h_flat \
            = np.moveaxis(m_grid, 0, -1).reshape(self.n_states_actions, self.d + self.d)

    def get_hjb_solver(self, h_hjb=0.001):
        from sde_hjb_solver.controlled_sde_1d import DoubleWellFHT1D as SDE1D
        from sde_hjb_solver.hjb_solver_1d_st import SolverHJB1D

        # initialize controlled sde object
        sde = SDE1D(
            beta=self.beta,
            alpha=self.alpha,
            domain=(-2,  2),
            target_set=(1, 2),
        )

        # initialize hjb solver object
        sol_hjb = SolverHJB1D(sde, h=1e-2)

        # load  hjb solver
        sol_hjb.load()

        # if hjb solver has different discretization step coarse solution
        if sol_hjb.sde.h < self.h_state:

            # discretization step ratio
            k = int(self.h_state / sol_hjb.sde.h)
            assert self.state_space_h.shape == sol_hjb.u_opt[::k].shape, ''

            sol_hjb.value_function = sol_hjb.value_function[::k]
            sol_hjb.u_opt = sol_hjb.u_opt[::k]

        return sol_hjb

    def get_det_policy_indices_from_hjb(self, policy_opt):
        return self.get_action_idx(np.expand_dims(policy_opt, axis=1))
