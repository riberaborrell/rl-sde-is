import numpy as np

def simple_lr_schedule(it, lr_init=1e-5, lr_final=1e-3, n_iter=1000):
    """
    Compute the learning rate at a given iteration using a logarithmic schedule.

    Parameters:
    - it: Current iteration (1-based index).
    - lr_init: Initial learning rate (default: 1e-5).
    - lr_final: Final learning rate (default: 1e-3).
    - n_iter: Total number of iterations (default: 1000).

    Returns:
    - Learning rate at the given iteration.
    """

    it = min(it, n_iter-1)
    final_log_scale_factor = np.log10(lr_final) - np.log10(lr_init)

    #scale_factor = 0 + (it - 1) * (final_log_scale_factor / (n_iter - 1))
    #return lr_init * 10 ** scale_factor
    return np.logspace(0, final_log_scale_factor, n_iter)[1:][it-1]


