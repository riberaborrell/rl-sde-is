import functools

import numpy as np
import scipy.stats as stats
import torch
from sde_hjb_solver.hjb_solver_1d_st import SolverHJB1D

from rl_sde_is.functions import *
from rl_sde_is.utils_numeric import *
from rl_sde_is.environments import SdeIsEnv

class SdeIs1DEnv(SdeIsEnv):
    '''
    '''

    def __init__(self, **kwargs):

        # dimension
        kwargs.update(d=1)

        super().__init__(**kwargs)


    def discretize_state_space(self, h_state):

        # discretize state space
        self.state_space_h = np.around(
            np.arange(
                self.state_space_bounds[0],
                self.state_space_bounds[1] + h_state,
                h_state,
            ),
            decimals=3,
        )
        self.n_states = self.state_space_h.shape[0]
        self.h_state = h_state

    def discretize_action_space(self, h_action):

        # discretize action space
        self.action_space_h = np.arange(
            self.action_space_bounds[0],
            self.action_space_bounds[1] + h_action,
            h_action,
        )
        self.n_actions = self.action_space_h.shape[0]
        self.h_action = h_action

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
        state = np.clip(state, self.state_space_bounds[0], self.state_space_bounds[1])
        idx = np.floor((state - self.state_space_bounds[0]) / self.h_state).astype(int)
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
        action = np.clip(action, self.action_space_bounds[0], self.action_space_bounds[1])
        idx = np.floor((action - self.action_space_bounds[0]) / self.h_action).astype(int)
        return idx[:, 0]

    def get_action_idx_min(self, action):
        return np.argmin(np.abs(self.action_space_h - action), axis=1)

    def get_state_init_idx(self):
        self.state_init_idx = self.get_state_idx(self.state_init)

    def get_target_set_idx_mgf(self):
        lb, rb = self.target_set
        self.is_in_ts = (self.state_space_h >= lb) \
                      & (self.state_space_h <= rb)
        self.lb_idx = self.get_state_idx(np.array([[lb]]))[0]
        self.rb_idx = self.get_state_idx(np.array([[rb]]))[0]
        self.ts_idx = np.where(self.is_in_ts)[0]
        self.not_ts_idx = np.where(np.invert(self.is_in_ts))[0]

    def get_target_set_idx_committor(self):
        #self.idx_a = np.where(self.state_space_h <= self.a_rb)[0]
        #self.idx_b = np.where(self.state_space_h >= self.b_lb)[0]
        #self.idx_ts = np.where((self.state_space_h <= self.a_rb) | (self.state_space_h >= self.b_lb))[0]
        #self.idx_not_ts = np.where((self.state_space_h > self.a_rb) & (self.state_space_h < self.b_lb))[0]

        self.is_in_a = (self.state_space_h >= self.target_set_a[0]) & (self.state_space_h <= self.target_set_a[1])
        self.is_in_b = (self.state_space_h >= self.target_set_b[0]) & (self.state_space_h <= self.target_set_b[1])
        self.is_in_ts = self.is_in_a | self.is_in_b

        # indices of domain_h in tha target set A and B and the target set
        self.ts_a_idx = np.where(self.is_in_a)[0]
        self.ts_b_idx = np.where(self.is_in_b)[0]
        self.ts_idx = np.where(self.is_in_ts)[0]
        self.not_ts_idx = np.where(np.invert(self.is_in_ts))[0]
        #self.idx_ts = np.where(
        #    ((self.state_space_h >= self.target_set_a[0]) & (self.state_space_h <= self.target_set_a[1])) |
        #    ((self.state_space_h >= self.target_set_b[0]) & (self.state_space_h <= self.target_set_b[1]))
        #)[0]

        # indices of the discretized domain corresponding to the target set
        #self.idx_not_ts = np.where(
        #    ((self.state_space_h < self.target_set_a[0]) | (self.state_space_h > self.target_set_a[1])) &
        #    ((self.state_space_h < self.target_set_b[0]) | (self.state_space_h > self.target_set_b[1]))
        #)[0]


    def get_null_action_idx(self):
        self.null_action_idx = self.get_action_idx(np.zeros((1, self.action_space_dim)))

    def get_greedy_actions(self, q_table):

        # compute greedy action by following the q-table
        actions_idx = np.argmax(q_table, axis=1)
        greedy_actions = self.action_space_h[actions_idx]

        # set actions in the target set to 0
        greedy_actions[self.ts_idx] = 0.
        return greedy_actions

    def discretize_state_action_space(self):

        # slice according to the state space and action space bounds and discretization step
        slice_i_state = slice(self.state_space_bounds[0], self.state_space_bounds[1] + self.h_state, self.h_state)
        slice_i_action = slice(self.action_space_bounds[0], self.action_space_bounds[1] + self.h_action, self.h_action)

        # number of state-actions pairs in the grid
        self.n_states_actions = self.n_states * self.n_actions

        # get state action space grid
        m_grid = np.mgrid[[
            slice_i_state,
            slice_i_action,
        ]]
        self.state_action_space_h_flat \
            = np.moveaxis(m_grid, 0, -1).reshape(self.n_states_actions, self.state_space_dim + self.action_space_dim)

    def get_det_policy_indices_from_hjb(self, policy_opt):
        return self.get_action_idx(np.expand_dims(policy_opt, axis=1))

    def reset(self, batch_size=1):
        if self.state_init_dist == 'delta':
            return np.full((batch_size, self.d), self.state_init)
        elif self.state_init_dist == 'uniform':
            return np.full(
                (batch_size, self.d),
                np.random.uniform(self.state_space_bounds[0], self.target_set[0], (self.d,))
            )
            #return np.full((batch_size, self.d), np.random.uniform(self.state_space_bounds[1], self.state_space_bounds[1], (self.d,)))

    def reset_done(self, states, done):
        if done.any():
            idx = np.where(done)[0]
            states[idx] = self.reset(batch_size=done.sum())

    def sample_state(self, batch_size=1):
        return np.random.uniform(self.state_space_bounds[0], self.state_space_bounds[1], (batch_size, self.d))

    def sample_state_ts(self, batch_size=1):
        return np.random.uniform(self.lb, self.state_space_bounds[1], (batch_size, self.d))

    def sample_action(self, batch_size=1):
       return np.random.uniform(
           self.action_space_bounds[0],
           self.action_space_bounds[1],
           (batch_size, self.d),
       )

    def state_action_transition_function(self, next_states, state, action, h):

        # compute mean and standard deviation
        mu = state + (- self.gradient_fn(state) + self.sigma * action) * self.dt
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


    def reward_signal_state_action_next_state_baseline(self, state, action, next_state, done):
        dist_fn = lambda x: np.linalg.norm(x - self.target_set[0], axis=1)
        running_cost = (self.f(state) + 0.5 * np.linalg.norm(action, axis=1)**2) * self.dt
        reward = np.where(
            done,
            - running_cost - self.g(next_state) + dist_fn(state),
            - running_cost + dist_fn(state) - dist_fn(next_state),
        )
        return reward


    def reward_signal_state_action_next_state_baseline_torch(self, state, action, next_state, done):
        dist_fn = lambda x: torch.linalg.norm(x - self.target_set[0], axis=1)
        running_cost = (self.f_torch(state) + 0.5 * torch.linalg.norm(action, axis=1)**2) * self.dt_tensor
        reward = torch.where(
            done,
            - running_cost - self.g_torch(next_state) + dist_fn(state),
            - running_cost + dist_fn(state) - dist_fn(next_state)
        )
        return reward



class OverdampedLangevinSDE1DEnv(SdeIs1DEnv):
    '''
    '''

    def __init__(self, beta=1., **kwargs):
        super().__init__(**kwargs)

        # overdamped langevin flag
        self.is_overdamped_langevin = True

        # inverse temperature
        self.beta = beta

        # diffusion
        self.sigma = np.sqrt(2 / self.beta)
        self.sigma_tensor = torch.tensor(self.sigma, dtype=tf32)


class DoubleWell1DEnv(OverdampedLangevinSDE1DEnv):
    ''' Overdamped langevin dynamics with double well potential.
    '''
    def __init__(self, alpha=1., **kwargs):
        super().__init__(**kwargs)

        # potential
        self.alpha = alpha
        self.potential_fn = functools.partial(double_well_1d, alpha=self.alpha)
        self.gradient_fn = functools.partial(double_well_gradient_1d, alpha=self.alpha)

        # drift term
        #self.drift = lambda x: - self.gradient_fn(x)

        # state and action space
        if self.state_space_bounds is None:
            self.state_space_bounds = (-2, 2)

        if self.action_space_bounds is None:
            breakpoint()
            self.set_action_space_bounds()


class DoubleWellMGF1DEnv(DoubleWell1DEnv):

    def __init__(self, lam=1.0, target_set=None, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'doublewell-1d-mgf__beta{:.1f}_alpha{:.1f}'.format(self.beta, self.alpha)

        # initial state
        self.state_init = np.array([[-1.]], dtype=nf32)

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = (1, 2)

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        self.is_in_target_set = lambda x: (x >= self.target_set[0]) & (x <= self.target_set[1])

    def set_action_space_bounds(self):
        if self.alpha == 1. and self.beta == 1:
            a = 3
        elif self.alpha == 1. and self.beta == 4:
            a = 5
        elif self.alpha == 1. and self.beta == 8:
            a = 7.5
        elif self.alpha == 5. and self.beta == 1:
            a = 8
        elif self.alpha == 10. and self.beta == 1:
            a = 20
        else:
            return
        self.action_space_bounds = - a, a

    def get_hjb_solver(self, h_hjb=0.001):
        from sde_hjb_solver.controlled_sde_1d import DoubleWellMGF1D as SDE1D

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


class DoubleWellCommittor1DEnv(DoubleWell1DEnv):

    def __init__(self, epsilon=1e-10, target_set_a=None, target_set_b=None, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'doublewell-1d-committor__beta{:.1f}_alpha{:.1f}'.format(self.beta, self.alpha)

        # initial state
        self.state_init = np.array([[0.]], dtype=nf32)

        # target set
        if target_set_a is not None:
            self.target_set_a = target_set_a
        else:
            self.target_set_a = (-2, -1)

        if target_set_b is not None:
            self.target_set_b = target_set_b
        else:
            self.target_set_b = (1, 2)

        # committor setting
        self.set_committor_setting(epsilon)

        # set in target set condition functions                                                    
        self.is_in_target_set_a = lambda x: (x >= self.target_set_a[0]) & (x <= self.target_set_a[1])
        self.is_in_target_set_b = lambda x: (x >= self.target_set_b[0]) & (x <= self.target_set_b[1])

    def set_action_space_bounds(self):
        if self.alpha == 1. and self.beta == 1:
            a = 20
        elif self.alpha == 1. and self.beta == 4:
            a = 5
        else:
            return
        self.action_space_bounds = 0, a
        breakpoint()

    def get_hjb_solver(self, h_hjb=0.001):
        from sde_hjb_solver.controlled_sde_1d import DoubleWellCommittor1D as SDE1D

        # initialize controlled sde object
        sde = SDE1D(
            beta=self.beta,
            alpha=self.alpha,
            domain=(-2,  2),
            target_set_a=(-2, -1),
            target_set_b=(1, 2),
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
