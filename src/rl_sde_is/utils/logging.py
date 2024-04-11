import numpy as np

def compute_array_statistics(array: np.array):
    mean = np.mean(array)
    var = np.var(array)
    re = np.sqrt(var) / mean
    return mean, var, re

def log_array_statistics(array: np.array, name: str = 'x'):
    mean, var, re = compute_array_statistics(array)
    print('{}: mean: {:2.4e}, var: {:2.4e}, re: {:.3f}'.format(name, mean, var, re))
