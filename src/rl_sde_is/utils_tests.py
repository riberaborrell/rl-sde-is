from collections import namedtuple

import numpy as np

from rl_sde_is.approximate_methods import test_policy_vectorized

Test = namedtuple('Test', ['freq_iterations', 'batch_size', 'mean_returns', 'var_returns',
                            'mean_lengths', 'policy_l2_error'])

class TestPolicy:
    def __init__(self, batch_size, freq=10, freq_type=None):

        # batch size
        self.batch_size = batch_size

        # frequency
        self.freq = freq
        if freq_type not in ['it', 'ep', 'steps']: raise ValueError()
        self.freq_type = freq_type
        self.counter = 0

        # arrays
        self.mean_returns =  np.empty((0), dtype=np.float32)
        self.var_returns = np.empty((0), dtype=np.float32)
        self.mean_lengths = np.empty((0), dtype=np.float32)
        self.policy_l2_errors = np.empty((0), dtype=np.float32)

    def test_actor_model(self, env, actor, policy_opt):
        mean_ret, var_ret, mean_len, policy_l2_error \
            = test_policy_vectorized(env, actor, batch_size=self.batch_size,
                                     policy_opt=policy_opt)

        # update arrays
        self.mean_returns = np.append(self.mean_returns, mean_ret)
        self.var_returns = np.append(self.var_returns, var_ret)
        self.mean_lengths = np.append(self.mean_lengths, mean_len)
        self.policy_l2_errors = np.append(self.policy_l2_errors, policy_l2_error)

        # print output
        msg = '{}: {:3d}, test mean return: {:2.2f}, test var return: {:.2e}, ' \
          'test mean time steps: {:2.2f}, test policy l2 error: {:.2e}'.format(
              self.freq_type,
              self.counter * self.freq,
              mean_ret,
              var_ret,
              mean_len,
              policy_l2_error,
          )
        print(msg)

        # update counter
        self.counter += 1

    def get_dict(self):
        return {
            'test_freq_type': self.freq_type,
            'test_freq': self.freq,
            'test_batch_size': self.batch_size,
            'test_mean_returns': self.mean_returns,
            'test_var_returns': self.var_returns,
            'test_mean_lengths': self.mean_lengths,
            'test_policy_l2_errors': self.policy_l2_errors,
        }

def preallocate_test_arrays(**kwargs):

    # mean, variance, mean length of the returns and l2 error after each epoch
    kwargs['test_mean_returns'] = np.empty((0), dtype=np.float32)
    """
    a = dict{
        'test_mean_returns':  np.empty((0), dtype=np.float32),
        'test_var_returns': np.empty((0), dtype=np.float32),
        'test_mean_lengths': np.empty((0), dtype=np.float32),
        'test_policy_l2_errors': np.empty((0), dtype=np.float32),
    }
    """
    return kwargs


