import numpy as np
import torch

from rl_sde_is.utils.replay_memory import ReplayMemory

class ReplayMemoryReturn(ReplayMemory):
    ''' replay memory for storing state-action pairs and n-returns or initial returns.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # initialize arrays and reset counters
        self.reset()

    def reset(self):

        # initialize state, action, n-returns and done arrays 
        self.states = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.actions = np.full((self.max_size, self.action_dim), np.nan, dtype=np.float32)
        self.returns = np.full(self.max_size, np.nan, dtype=np.float32)

        # counters and flags
        self.reset_counters()

    def store(self, state, action, ret):

        # update buffer arrays
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.n_returns[self.ptr] = ret
        self.update_store_idx_and_size()

    def store_vectorized(self, states, actions, returns):
        n_experiences = states.shape[0]
        i = self.ptr
        j = self.ptr + n_experiences
        assert j < self.max_size, 'The memory size is too low'

        self.states[i:j] = states
        self.actions[i:j] = actions
        self.returns[i:j] = returns

        self.ptr = (self.ptr + n_experiences) % self.max_size
        self.size = min(self.size + n_experiences, self.max_size)

    def sample_batch(self, batch_size, replace=True):

        # sample uniformly the batch indexes
        idx = self.sample_batch_idx(batch_size, replace)

        data = dict(
            states=self.states[idx],
            actions=self.actions[idx],
            returns=self.returns[idx]
        )
        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}

    #def estimate_episode_length(self):
    #    return self.size / self.done.sum()

