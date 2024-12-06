from typing import Optional, Union

import numpy as np
import torch
import scipy

nf32 = np.float32
ni32 = np.int32

tf32 = torch.float32
ti32 = torch.int32

def get_device(disable_cuda):
    if not disable_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# vectorized operations
def dot_vect(a, b):
    return (a * b).sum(axis=1)

def dot_vect_torch(a, b):
    return torch.matmul(
        torch.unsqueeze(a, dim=1),
        torch.unsqueeze(b, dim=2),
    ).squeeze()


# general logistic function
def logistic(x, L=1, k=1, x0=0):
        return L / (1 + np.exp(-k * (x - x0)))

def logistic_torch(x, L=1, k=1, x0=0):
        return L / (1 + torch.exp(-k * (x - x0)))

def compute_running_mean(x: np.array, run_window: Optional[int] = 10):
    ''' running mean / moving average of the array along the given running window.
    '''
    return np.array([
        np.mean(x[i-run_window:i+1]) if i > run_window
        else np.mean(x[:i+1]) for i in range(len(x))
    ])

def compute_running_variance(array: np.array, run_window: Optional[int] = 10):
    ''' running variance of the array along the given running window.
    '''
    return np.array([
        np.var(array[i-run_window:i+1]) if i > run_window
        else np.var(array[:i+1]) for i in range(len(array))
    ])

def cumsum_numpy(x):
    return x[::-1].cumsum()[::-1]

def cumsum_list(x):
    return cumsum_numpy(np.array(x))

def cumsum_torch(x):
    return torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])
    #return torch.cumsum(x.flip(dims=(0,)), dim=0).flip(dims=(0,))

def discount_cumsum_numpy(x, gamma):
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


#TODO: check if this is correct
def discount_cumsum_scipy(x, gamma):
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

def normalize_array(x: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-5):
    ''' Normalize the np.array or torch.tensor by subtracting the mean
        and dividing by the standard deviation.
    '''
    assert x.ndim == 1, 'Input must be a 1D array'
    return (x - x.mean()) / (x.std() + 1e-5)

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

def interpolate_array(xs, ys, n_points=100):

    assert xs.ndim == ys.ndim == 2, ''

    common_x = np.linspace(xs.min(axis=1).max(), xs.max(axis=1).min(), n_points)

    # interpolate y-array onto the common x grid
    interpolated_y = []
    for x, y in zip(xs, ys):
        interp_func = scipy.interpolate.interp1d(
            x, y, kind='linear', bounds_error=False, fill_value="extrapolate",
        )
        interpolated_y.append(interp_func(common_x))

    return common_x, np.array(interpolated_y)
