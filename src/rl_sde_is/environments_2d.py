import numpy as np
import scipy.stats as stats
import torch

class DoubleWellStoppingTime2D():

    def __init__(self, beta=1., alpha=1., dt=0.005, is_state_init_sampled=False):

        # environment log name
        self.name = 'doublewell-2d-st__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

        # double well potential
        self.d = 2
        #self.alpha = alpha * np.ones(self.d)
        self.alpha = np.full(self.d, alpha)
        self.alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32)

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
        self.state_space_dim = 2
        self.state_space_low = -2.
        self.state_space_high = 2.

        # action space bounds
        self.action_space_dim = 2
        self.action_space_low = 0.
        self.action_space_high = 3.

    def potential(self, state):
        return np.sum(self.alpha * (state**2 - 1) ** 2, axis=1)

    def potential_torch(self, state):
        return torch.sum(self.alpha * (state**2 - 1) ** 2, axis=1)

    def gradient(self, state):
        return 4 * self.alpha * state * (state**2 - 1)

    def gradient_torch(self, state):
        return 4 * self.alpha_tensor * state * (state**2 - 1)

    def is_done(self, state):
        return (state >= self.lb).all(axis=1)

    def is_done_torch(self, state):
        return (state >= self.lb).all(axis=1)
        #return torch.where(state >= self.lb, True, False)

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
            #return np.random.uniform(self.state_space_low, self.lb, (batch_size, self.d))
            return np.full((batch_size, self.d), np.random.uniform(self.state_space_low, self.lb, (self.d,)))

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
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt,
        )
        return reward, done

    def reward_signal_state_action_next_state(self, state, action, next_state):
        done = self.is_done(next_state)
        reward = np.where(
            done,
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt \
            - self.g(next_state),
            - (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt,
        )
        return reward, done

    def reward_signal_state_action_torch(self, state, action):
        done = self.is_done_torch(state)
        reward = torch.where(
            done,
            - self.g_torch(state),
            - (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor,
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

        # reward signal r_n = r(s_n, a_n)
        r, done = self.reward_signal_state_action(state, action)

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        #r, done = self.reward_signal_state_action_next_state(state, action, next_state)

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
                   + (- self.gradient_torch(state) + sigma * action) * dt \
                   + sigma * dbt

        # reward signal r_n = r(s_n, a_n)
        r, done = self.reward_signal_state_action_torch(state, action)

        # reward signal r_{n+1} = r(s_{n+1}, s_n, a_n)
        #r, done = self.reward_signal_state_action_next_state_torch(state, action, next_state)

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

    def get_state_idx(self, state):
        ''' get index of the corresponding discretized state
        '''
        # array input
        if state.ndim == 1:
            state = state[np.newaxis]

        return self.get_state_idx_truncate(state)

    def get_state_idx_truncate(self, state):
        state = np.clip(state, self.state_space_low, self.state_space_high)
        idx = np.floor((state - self.state_space_low) / self.h_state).astype(int)
        idx = tuple([idx[:, i] for i in range(self.d)])
        return idx

    def set_action_space_bounds(self):

        if self.alpha[0] == 1. and self.beta == 1:
            a = 3
        elif self.alpha[0] == 5. and self.beta == 1:
            a = 8
        elif self.alpha[0] == 1. and self.beta == 4:
            a = 5
        elif self.alpha[0] == 10. and self.beta == 1:
            a = 20
        else:
            return

        self.action_space_low = - a
        self.action_space_high = a

    def discretize_state_space(self, h_state):

        # slice according to the state space bounds and discretization step
        slice_i = slice(self.state_space_low, self.state_space_high + h_state, h_state)

        # get state space grid
        m_grid = np.mgrid[[slice_i, slice_i]]
        self.state_space_h = np.moveaxis(m_grid, 0, -1)

        # number of states and discretization step
        self.n_states_i1 = self.state_space_h.shape[0]
        self.n_states_i2 = self.state_space_h.shape[1]
        self.n_states = self.state_space_h.shape[0] * self.state_space_h.shape[1]
        self.h_state = h_state

        # get initial state index 
        self.get_idx_state_init()

    def get_idx_state_init(self):
        self.idx_state_init = self.get_state_idx(self.state_init)

    def get_action_idx(self, action):
        ''' get index of the corresponding discretized action
        '''
        # array input
        if action.ndim == 1:
            action = action[np.newaxis]

        return self.get_action_idx_truncate(action)

    def get_action_idx_truncate(self, action):
        action = np.clip(action, self.action_space_low, self.action_space_high)
        idx = np.floor((action - self.action_space_low) / self.h_action).astype(int)
        idx = tuple([idx[:, i] for i in range(self.d)])
        return idx

    def discretize_action_space(self, h_action):

        # slice according to the action space bounds and discretization step
        slice_i = slice(self.action_space_low, self.action_space_high + h_action, h_action)

        # get state space grid
        m_grid = np.mgrid[[slice_i, slice_i]]
        self.action_space_h = np.moveaxis(m_grid, 0, -1)

        # number of states and discretization step
        self.n_actions_i1 = self.action_space_h.shape[0]
        self.n_actions_i2 = self.action_space_h.shape[1]
        self.n_actions = self.action_space_h.shape[0] * self.action_space_h.shape[1]
        self.h_action = h_action

        # get null action index
        self.get_idx_null_action()

    def get_idx_null_action(self):
        self.idx_null_action = self.get_action_idx(np.zeros((1, self.d)))

    def discretize_state_action_space(self):

        # slice according to the state space and action space bounds and discretization step
        slice_i_state = slice(self.state_space_low, self.state_space_high + self.h_state, self.h_state)
        slice_i_action = slice(self.action_space_low, self.action_space_high + self.h_action, self.h_action)

        # number of state-actions pairs in the grid
        self.n_states_actions = self.n_states * self.n_actions

        # get state action space grid
        m_grid = np.mgrid[[
            slice_i_state,
            slice_i_state,
            slice_i_action,
            slice_i_action,
        ]]
        self.state_action_space_h_flat \
            = np.moveaxis(m_grid, 0, -1).reshape(self.n_states_actions, self.d * self.d)

    def get_hjb_solver(self, h_hjb=0.001):
        from sde_hjb_solver.controlled_sde_2d import DoubleWellStoppingTime2D as SDE2D
        from sde_hjb_solver.hjb_solver_2d_st import SolverHJB2D

        # initialize controlled sde object
        sde = SDE2D(
            beta=self.beta,
            alpha=self.alpha,
            domain=(-2,  2),
            target_set=(1, 2),
        )

        # initialize hjb solver object
        sol_hjb = SolverHJB2D(sde, h=1e-1)

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
        policy_opt_flat = policy_opt.reshape(self.n_states, self.d)
        return np.vstack(self.get_action_idx(policy_opt_flat)).T
