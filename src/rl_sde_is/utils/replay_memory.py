from typing import Optional

import numpy as np

class ReplayMemory:
    def __init__(self, size: int, state_dim: int,
                 action_dim: Optional[int] = None, is_action_continuous: bool = True):

        # memory parameters
        self.max_size = size
        self.state_dim = state_dim
        self.is_action_continuous = is_action_continuous
        if is_action_continuous:
            assert action_dim is not None, ''
            self.action_dim = action_dim

    def reset_counters(self):
        ''' reset counters and flags'''
        self.ptr = 0
        self.size = 0
        self.is_full = False

    def update_store_idx_and_size(self):
        ''' update the store index and size of the memory replay'''
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        if not self.is_full and self.size == self.max_size:
            self.is_full = True
            print('Replay memory is full!')

    def sample_batch_idx(self, batch_size: int, replace: bool = True) -> np.ndarray:
        ''' sample uniformly the batch indices'''
        return np.random.choice(self.size, size=batch_size, replace=replace)

