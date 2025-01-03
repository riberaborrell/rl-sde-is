import numpy as np

def simple_lr_schedule(it, lr_init=1e-5, lr_final=1e-3, n_iter=1000):

    final_log_scale_factor = np.log10(lr_final) - np.log10(lr_init)

    return np.logspace(0, final_log_scale_factor, n_iter)[it-1]

def two_phase_lr_schedule(it, lr_init=1e-5, lr_final=1e-3, n_iter_adaptive=1000):

    final_log_scale_factor = np.log10(lr_final) - np.log10(lr_init)

    # increasing or decreasing phase (from lr_init to lr_final)
    if it <= n_iter_adaptive:
        return np.logspace(0, final_log_scale_factor, n_iter_adaptive)[it-1]

    # constant phase
    else:
        return 10**final_log_scale_factor

def three_phase_lr_schedule(it, lr_init=1e-5, lr_middle=1e-2, lr_final=1e-3,
                            n_iter_1=1000, n_iter_2=1000):

    middle_log_scale_factor = np.log10(lr_middle) - np.log10(lr_init)
    final_log_scale_factor = np.log10(lr_final) - np.log10(lr_init)

    # increase phase (from lr_init to lr_max)
    if it <= n_iter_1:
        return np.logspace(0, middle_log_scale_factor, n_iter_1)[it-1]

    # decrease phase (from lr_max to lr_final)
    elif n_iter_1 < it <= n_iter_1 + n_iter_2:
        return np.logspace(middle_log_scale_factor, final_log_scale_factor, n_iter_2)[it-n_iter_1-1]

    # constant phase
    else:
        return 10**final_log_scale_factor
