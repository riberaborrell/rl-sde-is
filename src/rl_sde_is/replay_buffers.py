import numpy as np

class DiscreteReplayBuffer:

    def __init__(self, state_dim, size):
        self.state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.idx_act_buf = np.zeros(size, dtype=np.int64)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.bool)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, state, idx_act, rew, next_state, done):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.idx_act_buf[self.ptr] = idx_act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(state=self.state_buf[idxs],
                    next_state=self.next_state_buf[idxs],
                    idx_act=self.idx_act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])

class ContinuousReplayBuffer:

    def __init__(self, state_dim, action_dim, size):
        self.state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.bool)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, state, action, rew, next_state, done):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(state=self.state_buf[idxs],
                    next_state=self.next_state_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])
