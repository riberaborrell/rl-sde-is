import numpy as np
import torch

def compute_running_mean(array, run_window=10):
    ''' computes the running mean / moving average of the array along the given running window.
    '''
    return np.array([
        np.mean(array[i-run_window:i+1]) if i > run_window
        else np.mean(array[:i+1]) for i in range(len(array))
    ])

def compute_running_variance(array, run_window=10):
    ''' computes the running variance of the array along the given running window.
    '''
    return np.array([
        np.var(array[i-run_window:i+1]) if i > run_window
        else np.var(array[:i+1]) for i in range(len(array))
    ])

def cumsum_list(x):
    x = np.array(x)
    return x[::-1].cumsum()[::-1]

def cumsum_numpy(x):
    return x[::-1].cumsum()[::-1]

def cumsum_torch(x):
    return torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])
    #return torch.cumsum(x.flip(dims=(0,)), dim=0).flip(dims=(0,))

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def discount_cumsum_torch(x, gamma):
    n = x.shape[0]
    y = gamma**torch.arange(n)
    z = torch.zeros_like(x, dtype=torch.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z


def discount_cumsum_torch_scipy(x, gamma):
    import scipy
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
    input:
        vector x,
        [x0,
         x1,
         x2]

     output:
        [x0 + gamma * x1 + gamma^2 * x2,
         x1 + gamma * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]

def sample_items_original(prob_matrix, items):
    n = prob_matrix.shape[1]
    choices = np.zeros((n,))
    for i in range(n):
        choices[i] = np.random.choice(items, p=prob_matrix[:, i])
    return choices

def sample_items(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]
