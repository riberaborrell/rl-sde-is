import time

import numpy as np
import pytest

from rl_sde_is.environments import DoubleWellStoppingTime1D
from rl_sde_is.environments_2d import DoubleWellStoppingTime2D


class TestEnvironments:

    @pytest.fixture
    def env_dwst1d(self, alpha, beta):
        '''
        '''
        env = DoubleWellStoppingTime1D(
            alpha=alpha,
            beta=beta,
        )
        return env

    @pytest.fixture
    def env_dwst2d(self, alpha, beta):
        '''
        '''
        env = DoubleWellStoppingTime2D(
            alpha=alpha,
            beta=beta,
        )
        return env

    def test_discretize_state_space_1d(self, env_dwst1d, h_state):
        ''' test state space discretization.
        '''
        env = env_dwst1d

        # discretize domain
        env.discretize_state_space(h_state)

        # compute size of domain
        state_space_h_size = np.around(env.state_space_h.nbytes / 1024 / 1024, 2)
        assert env.n_states <= int(1e6)

    def test_get_state_idx_1d(self, env_dwst1d, h_state, batch_size):
        env = env_dwst1d

        # discretize domain
        env.discretize_state_space(h_state)

        # sample point uniformly
        states = env.sample_state(batch_size)

        # get states index and evaluate in the grid
        idx = env.get_state_idx(states)
        states_h = np.expand_dims(env.state_space_h[idx], axis=1)

        assert np.isclose(states, states_h, atol=h_state).all()

    def test_get_action_idx_1d(self, env_dwst1d, h_action, batch_size):
        env = env_dwst1d

        # discretize domain
        env.discretize_action_space(h_action)

        # sample point uniformly
        actions = env.sample_action(batch_size)

        # get index
        idx = env.get_action_idx(actions)

        actions_h = np.expand_dims(env.action_space_h[idx], axis=1)

        assert np.isclose(actions, actions_h, atol=h_action).all()

    def test_get_state_idx_2d(self, env_dwst2d, h_state, batch_size):
        env = env_dwst2d

        # discretize domain
        env.discretize_state_space(h_state)

        # sample point uniformly
        states = env.sample_state(batch_size)

        # get states index and evaluate in the grid
        idx = env.get_state_idx(states)
        states_h = env.state_space_h[idx]

        assert np.isclose(states, states_h, atol=h_state).all()
