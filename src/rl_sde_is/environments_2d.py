import functools

import numpy as np
import scipy.stats as stats
import torch
from sde_hjb_solver.hjb_solver_2d_st import SolverHJB2D

from rl_sde_is.functions import *
from rl_sde_is.utils_numeric import *
from rl_sde_is.environments import SdeIsEnv

class SdeIs2DEnv(SdeIsEnv):
    '''
    '''

    def __init__(self, **kwargs):

        # dimension
        kwargs.update(d=2)

        super().__init__(**kwargs)

    def discretize_state_space(self, h_state):

        # slice according to the state space bounds and discretization step
        slice_i = slice(
            self.state_space_bounds[0],
            self.state_space_bounds[1] + h_state,
            h_state,
        )

        # get state space grid
        m_grid = np.mgrid[[slice_i, slice_i]]
        self.state_space_h = np.moveaxis(m_grid, 0, -1)

        # number of states and discretization step
        self.n_states_i1 = self.state_space_h.shape[0]
        self.n_states_i2 = self.state_space_h.shape[1]
        self.n_states = self.state_space_h.shape[0] * self.state_space_h.shape[1]
        self.h_state = h_state

        # get initial state index 
        #self.get_state_init_idx()

    def discretize_action_space(self, h_action):

        # slice according to the action space bounds and discretization step
        slice_i = slice(
            self.action_space_bounds[0],
            self.action_space_bounds[1] + h_action,
            h_action,
        )

        # get state space grid
        m_grid = np.mgrid[[slice_i, slice_i]]
        self.action_space_h = np.moveaxis(m_grid, 0, -1)

        # number of states and discretization step
        self.n_actions_i1 = self.action_space_h.shape[0]
        self.n_actions_i2 = self.action_space_h.shape[1]
        self.n_actions = self.action_space_h.shape[0] * self.action_space_h.shape[1]
        self.h_action = h_action

        # get null action index
        #self.get_null_action_idx()

    def get_state_idx(self, state):
        ''' get index of the corresponding discretized state
        '''
        # array input
        if state.ndim == 1:
            state = state[np.newaxis]

        return self.get_state_idx_truncate(state)

    def get_state_idx_truncate(self, state):
        state = np.clip(state, self.state_space_bounds[0], self.state_space_bounds[1])
        idx = np.floor((state - self.state_space_bounds[0]) / self.h_state).astype(int)
        idx = tuple([idx[:, i] for i in range(self.d)])
        return idx

    def get_state_init_idx(self):
        self.state_init_idx = self.get_state_idx(self.state_init)

    def get_target_set_idx_mgf(self):
        pass

    def get_target_set_idx_committor(self):
        pass

    def get_action_idx(self, action):
        ''' get index of the corresponding discretized action
        '''
        # array input
        if action.ndim == 1:
            action = action[np.newaxis]

        return self.get_action_idx_truncate(action)

    def get_action_idx_truncate(self, action):
        action = np.clip(action, self.action_space_bounds[0], self.action_space_bounds[1])
        idx = np.floor((action - self.action_space_bounds[0]) / self.h_action).astype(int)
        idx = tuple([idx[:, i] for i in range(self.d)])
        return idx

    def get_null_action_idx(self):
        self.null_action_idx = self.get_action_idx(np.zeros((1, self.d)))

    def reset(self, batch_size=1):
        if self.state_init_dist == 'delta':
            return np.full((batch_size, self.d), self.state_init)
        elif self.state_init_dist == 'uniform':
            #return np.random.uniform(self.state_space_bounds[0], self.lb, (batch_size, self.d))
            return np.full(
                (batch_size, self.d),
                np.random.uniform(self.state_space_bounds[0], self.lb, (self.d,))
            )

    def sample_state(self, batch_size=1):
            return np.random.uniform(self.state_space_bounds[0], self.state_space_bounds[1], (batch_size, self.d))

    def sample_action(self, batch_size=1):
            return np.random.uniform(self.action_space_bounds[0], self.action_space_bounds[1], (batch_size, self.d))


class OverdampedLangevinSDE2DEnv(SdeIs2DEnv):
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


class DoubleWell2DEnv(OverdampedLangevinSDE2DEnv):
    ''' Overdamped langevin dynamics with double well potential.
    '''
    def __init__(self, alpha=1., **kwargs):
        super().__init__(**kwargs)

        # potential
        self.alpha = alpha
        self.potential_fn = functools.partial(double_well_nd, alpha=self.alpha)
        self.gradient_fn = functools.partial(double_well_gradient_nd, alpha=self.alpha)

        # drift term
        #self.drift = lambda x: - self.gradient_fn(x)

        # state and action space
        if self.state_space_bounds is None:
            self.state_space_bounds = (-2, 2)

        if self.action_space_bounds is None:
            self.set_action_space_bounds()

class DoubleWellMGF2DEnv(DoubleWell2DEnv):

    def __init__(self, lam=1.0, target_set=None, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'doublewell-2d-mgf__beta{:.1f}_alpha{:.1f}'.format(self.beta, self.alpha)

        # initial state
        self.state_init = - np.ones((1, self.d), dtype=np.float32)

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = (1, 2)

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        #self.is_in_target_set = lambda x: (x >= self.target_set[0]) & (x <= self.target_set[1])
        self.is_in_target_set = lambda x: (x[:, 0] >= self.target_set[0]) \
                                        & (x[:, 0] <= self.target_set[1]) \
                                        & (x[:, 1] >= self.target_set[0]) \
                                        & (x[:, 1] <= self.target_set[1])


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
        from sde_hjb_solver.controlled_sde_2d import DoubleWellMGF2D as SDE2D

        # initialize controlled sde object
        sde = SDE2D(
            beta=self.beta,
            alpha=np.full(self.d, self.alpha),
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
