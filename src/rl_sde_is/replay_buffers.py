from collections import namedtuple, deque
import random

import numpy as np


class ReplayBuffer:

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

        # initialize arrays
        self.states = np.zeros([self.max_size, self.state_dim], dtype=np.float32)
        self.next_states = np.zeros([self.max_size, self.state_dim], dtype=np.float32)

        if self.is_action_continuous:
            self.actions = np.zeros([self.max_size, self.action_dim], dtype=np.float32)
        else:
            self.actions = np.zeros(self.max_size, dtype=np.int64)

        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.done = np.zeros(self.max_size, dtype=bool)

        # counters and flags
        self.ptr = 0
        self.size = 0
        self.is_full = False

    def store(self, state, action, reward, next_state, done):

        # update buffer
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        if not self.is_full and self.size == self.max_size:
            self.is_full = True
            print('Replay buffer is full!')

    def store_vectorized(self, states, actions, rewards, next_states, done):
        n_transitions = states.shape[0]
        i = self.ptr
        j = self.ptr + n_transitions

        self.states[i:j] = states
        self.actions[i:j] = actions
        self.rewards[i:j] = rewards
        self.next_states[i:j] = next_states
        self.done[i:j] = done

        self.ptr = (self.ptr + n_transitions) % self.max_size
        self.size = min(self.size + n_transitions, self.max_size)

        #if not self.is_full and self.size == self.max_size:
        #    self.is_full = True
        #    print('Replay buffer is full!')


    def sample_batch(self, batch_size=None):

        # sample the whole replay buffer
        if batch_size == None:
            idxs = np.arange(self.size)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(states=self.states[idxs],
                    actions=self.actions[idxs],
                    rewards=self.rewards[idxs],
                    next_states=self.next_states[idxs],
                    done=self.done[idxs])

    def estimate_episode_length(self):
        return self.done.sum() / self.size


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'done')
)

class ReplayMemory(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = deque([], maxlen=max_size)

    def reset(self):
        self.memory = deque([], maxlen=self.max_size)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)
