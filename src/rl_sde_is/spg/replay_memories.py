import numpy as np
import torch

class ReplayMemoryReturn:

    def __init__(self, size, state_dim, action_dim, return_type='n-return'):

        assert return_type in ['n-return', 'initial-return'], 'Invalid return type'

        # buffer parameters
        self.max_size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.return_type = return_type

        # initialize arrays and reset counters
        self.reset()

    def reset(self):

        # initialize state, action, n-returns and done arrays 
        self.states = np.full((self.max_size, self.state_dim), np.nan, dtype=np.float32)
        self.actions = np.full((self.max_size, self.action_dim), np.nan, dtype=np.float32)
        if self.return_type == 'n-return':
            self.n_returns = np.full(self.max_size, np.nan, dtype=np.float32)
        else: # self.return_type == 'initial-return':
            self.initial_returns = np.full(self.max_size, np.nan, dtype=np.float32)

        # counters and flags
        self.ptr = 0
        self.size = 0
        self.is_full = False

    def store(self, state, action, n_return=None, initial_return=None):

        # update buffer arrays
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        if self.return_type == 'n-return':
            self.n_returns[self.ptr] = n_return
        else:
            self.initial_returns[self.ptr] = initial_return

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        if not self.is_full and self.size == self.max_size:
            self.is_full = True
            print('Replay buffer is full!')

    def store_vectorized(self, states, actions, n_returns=None, initial_returns=None):
        n_experiences = states.shape[0]
        i = self.ptr
        j = self.ptr + n_experiences
        assert j < self.max_size, 'The memory size is too low'

        self.states[i:j] = states
        self.actions[i:j] = actions
        if self.return_type == 'n-return':
            self.n_returns[i:j] = n_returns
        else:
            self.initial_returns[i:j] = initial_returns

        self.ptr = (self.ptr + n_experiences) % self.max_size
        self.size = min(self.size + n_experiences, self.max_size)

    def sample_batch(self, batch_size, replace=True):

        # sample uniformly the batch indexes
        idx = np.random.choice(self.size, size=batch_size, replace=replace)

        data = dict(
            states=self.states[idx],
            actions=self.actions[idx],
        )
        if self.return_type == 'n-return':
            data['n-returns'] = self.n_returns[idx]
        else:
            data['initial-returns'] = self.initial_returns[idx]

        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}

    #def estimate_episode_length(self):
    #    return self.size / self.done.sum()

class ReplayMemoryIS:

    def __init__(self, size, state_dim, action_dim=None, is_action_continuous=True):

        # buffer parameters
        self.max_size = size
        self.state_dim = state_dim
        self.is_action_continuous = is_action_continuous
        if is_action_continuous:
            assert action_dim is not None, ''
            self.action_dim = action_dim

        # initialize arrays and reset counters
        self.reset()

    def reset(self):

        # initialize (s, a, r, s', d) arrays
        self.states = np.zeros([self.max_size, self.state_dim], dtype=np.float32)
        self.next_states = np.zeros([self.max_size, self.state_dim], dtype=np.float32)

        if self.is_action_continuous:
            self.actions = np.zeros([self.max_size, self.action_dim], dtype=np.float32)
        else:
            self.actions = np.zeros(self.max_size, dtype=np.int64)

        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.done = np.zeros(self.max_size, dtype=bool)

        # initialize importance sampling arrays
        self.is_on_policy = np.zeros(self.max_size, dtype=np.int32)
        self.importance_weights = np.zeros(self.max_size, dtype=np.float32)
        self.truncated_importance_weights = np.zeros(self.max_size, dtype=np.float32)
        self.behav_means = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.behav_stds = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.curr_means = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.curr_stds = np.zeros((self.max_size, self.action_dim), dtype=np.float32)

        # counters and flags
        self.ptr = 0
        self.size = 0
        self.is_full = False

    def store(self, state, action, reward, next_state, done, mean, std):

        # update (s, a, r, s', d) buffer arrays
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.done[self.ptr] = done

        # update importance sampling arrays
        self.is_on_policy[self.ptr] = True
        self.importance_weights[self.ptr] = 1.
        self.truncated_importance_weights[self.ptr] = 1.
        self.behav_means[self.ptr] = mean
        self.behav_stds[self.ptr] = std
        self.curr_means[:] = mean
        self.curr_stds[:] = std

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        if not self.is_full and self.size == self.max_size:
            self.is_full = True
            print('Replay buffer is full!')

    def sample_batch(self, batch_size, replace=True):

        # sample uniformly the batch indexes
        idx = np.random.choice(self.size, size=batch_size, replace=replace)

        return dict(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            done=self.done[idx],
            is_on_policy=self.is_on_policy[idx],
            importance_weights=self.importance_weights[idx],
            truncated_importance_weights=self.truncated_importance_weights[idx],
            behav_means=self.behav_means[idx],
            behav_stds=self.behav_stds[idx],
            curr_means=self.curr_means[idx],
            curr_stds=self.curr_stds[idx],
        )

    def estimate_episode_length(self):
        return self.size / self.done.sum()
