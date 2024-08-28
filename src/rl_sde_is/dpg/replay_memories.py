import numpy as np
import torch

from rl_sde_is.utils.numeric import cumsum_numpy as cumsum, discount_cumsum_scipy as discount_cumsum

class ReplayMemory:

    def __init__(self, size, state_dim, action_dim=None, is_action_continuous=True):

        # memory parameters
        self.max_size = size
        self.state_dim = state_dim
        self.is_action_continuous = is_action_continuous
        if is_action_continuous:
            assert action_dim is not None, ''
            self.action_dim = action_dim

        # initialize arrays and reset counters
        self.reset()

    def reset(self):

        # initialize arrays
        self.states = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.next_states = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)

        if self.is_action_continuous:
            self.actions = np.full((self.max_size, self.action_dim), np.nan, dtype=np.float32)
        else:
            self.actions = np.full(self.max_size, np.nan, dtype=np.int64)

        self.rewards = np.full(self.max_size, np.nan, dtype=np.float32)
        self.done = np.zeros(self.max_size, dtype=bool)

        # counters and flags
        self.ptr = 0
        self.size = 0
        self.is_full = False

    def store(self, state, action, reward, next_state, done):

        # update memory
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        if not self.is_full and self.size == self.max_size:
            self.is_full = True
            print('Replay memory is full!')

    def store_vectorized(self, states, actions, rewards, next_states, done):
        n_transitions = states.shape[0]
        i = self.ptr
        j = self.ptr + n_transitions
        if j > self.max_size:
            raise ValueError('Memory is full')

        self.states[i:j] = states
        self.actions[i:j] = actions
        self.rewards[i:j] = rewards
        self.next_states[i:j] = next_states
        self.done[i:j] = done

        self.ptr = (self.ptr + n_transitions) % self.max_size
        self.size = min(self.size + n_transitions, self.max_size)

        #if not self.is_full and self.size == self.max_size:
        #    self.is_full = True
        #    print('Replay memory is full!')


    def sample_batch(self, batch_size, replace=True):

        # sample uniformly the batch indices
        idx = np.random.choice(self.size, size=batch_size, replace=replace)

        data = dict(
            states=torch.as_tensor(self.states[idx], dtype=torch.float32),
            actions=torch.as_tensor(self.actions[idx], dtype=torch.float32),
            rewards=torch.as_tensor(self.rewards[idx], dtype=torch.float32),
            next_states=torch.as_tensor(self.next_states[idx], dtype=torch.float32),
            done=torch.as_tensor(self.done[idx], dtype=bool)
        )
        return data

    def estimate_episode_length(self):
        return self.size / self.done.sum()

class ReplayMemoryModelBasedDPG:

    def __init__(self, size, state_dim, gamma=1.0):

        # memory parameters
        self.max_size = size
        self.state_dim = state_dim
        self.gamma = gamma

        # initialize arrays and reset counters
        self.reset()

    def reset(self):

        # initialize arrays
        self.states = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.dbts = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.rewards = np.full(self.max_size, np.nan, dtype=np.float32)
        self.returns = np.full(self.max_size, np.nan, dtype=np.float32)
        self.lengths = np.full(self.max_size, np.nan, dtype=np.float32)

        # counters and flags
        self.ptr = 0
        self.size = 0
        self.path_start_idx = 0
        self.is_full = False

    def store(self, state, dbt, reward=None, ret=None):

        # update buffer
        self.states[self.ptr] = state
        self.dbts[self.ptr] = dbt
        if reward is not None:
            self.rewards[self.ptr] = reward

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        if not self.is_full and self.size == self.max_size:
            self.is_full = True
            print('Replay buffer is full!')

    def store_vectorized(self, states, dbts, rewards=None, returns=None):
        n_experiences = states.shape[0]
        i = self.ptr
        j = self.ptr + n_experiences
        assert j < self.max_size, 'The memory size is too low'

        # update buffer
        self.states[i:j] = states
        self.dbts[i:j] = dbts
        if rewards is not None:
            self.rewards[i:j] = rewards
        if returns is not None:
            self.returns[i:j] = returns

        self.ptr = (self.ptr + n_experiences) % self.max_size
        self.size = min(self.size + n_experiences, self.max_size)
        if not self.is_full and self.size == self.max_size:
            self.is_full = True
            print('Replay buffer is full!')

    def finish_path(self):
        if self.path_start_idx <= self.ptr:
            path_slice = slice(self.path_start_idx, self.ptr)
            rewards = self.rewards[path_slice]
            self.returns[path_slice] = rewards.sum()
            self.returns[path_slice] = discount_cumsum(rewards, self.gamma)#[:-1]
            self.lengths[path_slice] = rewards.shape[0]

        else:
            path_slice1 = slice(self.path_start_idx, None)
            path_slice2 = slice(0, self.ptr)
            rewards = np.append(self.rewards[path_slice1], self.rewards[path_slice2])
            initial_return = rewards.sum()
            #returns = discount_cumsum(rewards, self.gamma)[:-1]
            length = rewards.shape[0]
            self.returns[path_slice1], self.returns[path_slice2] = initial_return
            #self.returns[path_slice1] = returns[path_slice1]
            #self.returns[path_slice2] = returns[path_slice2]
            self.lengths[path_slice1], self.lengths[path_slice1] = length

        self.path_start_idx = self.ptr

    def sample_batch(self, batch_size, replace=True):

        # sample uniformly the batch indices
        idx = np.random.choice(self.size, size=batch_size, replace=replace)

        data = dict(
            states=self.states[idx],
            dbts=self.dbts[idx],
            returns=self.returns[idx],
            lengths=self.lengths[idx],
        )
        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}

    def estimate_mean_episode_length(self):
        return self.lengths.mean() if self.is_full else self.lengths[:self.ptr].mean()
