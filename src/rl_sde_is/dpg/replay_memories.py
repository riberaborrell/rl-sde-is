import numpy as np
import torch

from rl_sde_is.utils.numeric import cumsum_numpy as cumsum, discount_cumsum_scipy as discount_cumsum

class ReplayMemory:
    def __init__(self, size):

        # memory parameters
        self.max_size = size

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

    def sample_batch_idx(self, batch_size, replace=True):
        ''' sample uniformly the batch indices'''
        return np.random.choice(self.size, size=batch_size, replace=replace)


class ReplayMemoryModelFreeDPG(ReplayMemory):

    def __init__(self, state_dim, action_dim=None, is_action_continuous=True, **kwargs):

        super().__init__(**kwargs)

        # memory parameters
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
        self.reset_counters()

    def store(self, state, action, reward, next_state, done):

        # update memory
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.done[self.ptr] = done

        self.update_store_idx_and_size()

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

        idx = self.sample_batch_idx(batch_size, replace)
        data = dict(
            states=torch.as_tensor(self.states[idx], dtype=torch.float32),
            actions=torch.as_tensor(self.actions[idx], dtype=torch.float32),
            rewards=torch.as_tensor(self.rewards[idx], dtype=torch.float32),
            next_states=torch.as_tensor(self.next_states[idx], dtype=torch.float32),
            done=torch.as_tensor(self.done[idx], dtype=bool)
        )
        return data

    def estimate_mean_episode_length(self):
        return self.size / self.done.sum()

class ReplayMemoryModelBasedDPG(ReplayMemory):

    def __init__(self, state_dim=1, gamma=1.0, **kwargs):

        super().__init__(**kwargs)

        # memory parameters
        self.state_dim = state_dim
        self.gamma = gamma

        # initialize arrays and reset counters
        self.reset()

    def reset(self):

        # initialize arrays
        self.states = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.dbts = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.rewards = np.full(self.max_size, np.nan, dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=bool)
        self.returns = np.full(self.max_size, np.nan, dtype=np.float32)
        self.lengths = np.full(self.max_size, np.nan, dtype=np.float32)

        # counters and flags
        self.traj_start_idx = 0
        self.reset_counters()

    def store(self, state, dbt, reward=None, done=None, ret=None):

        # update memory
        self.states[self.ptr] = state
        self.dbts[self.ptr] = dbt
        if reward is not None:
            self.rewards[self.ptr] = reward
        if done is not None:
            self.dones[self.ptr] = done

        self.update_store_idx_and_size()

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

    def compute_returns(self, return_type='n-return'):

        # get trajectory rewards
        if self.traj_start_idx <= self.ptr:
            traj_slice = slice(self.traj_start_idx, self.ptr)
            rewards = self.rewards[traj_slice]
        else:
            traj_slice1 = slice(self.traj_start_idx, None)
            traj_slice2 = slice(0, self.ptr)
            rewards = np.append(self.rewards[traj_slice1], self.rewards[traj_slice2])
        traj_length = rewards.shape[0]

        # compute returns
        if return_type == 'n-return' and self.gamma == 1.:
            returns = np.append(cumsum(rewards)[1:], 0.)
        elif return_type == 'n-return' and self.gamma < 1.:
            returns = np.append(discount_cumsum(rewards, self.gamma)[1:], 0.)
        elif return_type == 'initial-return':
            returns = np.full(rewards.sum(), traj_length)

        # store returns and lengths
        if self.traj_start_idx <= self.ptr:
            self.lengths[traj_slice] = traj_length
            self.returns[traj_slice] = returns
        else:
            self.returns[traj_slice1] = returns[:self.size - self.traj_start_idx]
            self.returns[traj_slice2] = returns[self.size - self.traj_start_idx:]
            self.lengths[traj_slice1], self.lengths[traj_slice2] = traj_length, traj_length

        self.traj_start_idx = self.ptr

    def sample_batch(self, batch_size, replace=True):

        idx = self.sample_batch_idx(batch_size, replace)
        data = dict(
            states=torch.as_tensor(self.states[idx], dtype=torch.float32),
            dbts=torch.as_tensor(self.dbts[idx], dtype=torch.float32),
            dones=torch.as_tensor(self.dones[idx], dtype=bool),
            returns=torch.as_tensor(self.returns[idx], dtype=torch.float32),
            lengths=torch.as_tensor(self.lengths[idx], dtype=torch.int32)
        )
        return data

    def estimate_mean_episode_length(self):
        return self.lengths.mean() if self.is_full else self.lengths[:self.ptr].mean()
