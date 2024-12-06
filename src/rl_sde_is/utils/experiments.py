import numpy as np

from rl_sde_is.dpg.reinforce_deterministic_core import reinforce_deterministic
from rl_sde_is.spg.reinforce_stochastic_core import reinforce_stochastic
from rl_sde_is.utils.numeric import compute_running_mean

def get_coarse_lrs(lr_low, lr_high):
    assert lr_low < lr_high, ''
    assert np.log10(lr_low) % 1 == 0, ''
    assert np.log10(lr_high) % 1 == 0, ''

    e_low, e_high = int(np.log10(lr_low)), int(np.log10(lr_high))
    lrs = []
    for e in range(e_low, e_high+1):
        lrs.append(10**(e))
    return lrs

def get_fine_lrs(lr_low, lr_high):
    assert lr_low < lr_high, ''
    assert np.log10(lr_low) % 1 == 0, ''
    assert np.log10(lr_high) % 1 == 0, ''

    e_low, e_high = int(np.log10(lr_low)), int(np.log10(lr_high))
    lrs = []
    for e in range(e_low, e_high):
        lrs.append(10**(e))
        lrs.append(2*10**(e))
        lrs.append(5*10**(e))
    lrs.append(10**(e_high))
    return lrs

def get_arrays_multiple_datas(datas: list[dict], keys: list[str]) -> dict:
    ''' Get the values of the given keys from multiple dictionaries.
    '''
    assert len(datas) > 0, 'No data is provided'
    for data in datas:
        for key in keys:
            assert key in data.keys(), 'The given key is not in the dictionary'

    # dictionary containing the chosen arrays 
    res = {key: [] for key in keys}

    for key in keys:
        for i, data in enumerate(datas):
            res[key].append(data[key])
        res[key] = np.vstack(res[key])

    return res


def get_cum_arrays_multiple_datas(datas: list[dict], keys: list[str]) -> dict:

    res = get_arrays_multiple_datas(datas, keys)
    for key in keys:
        res[key] = np.hstack((np.zeros((res[key].shape[0], 1)), res[key].cumsum(axis=1)))
    return res


def get_n_iterations_until_goal(data, key, threshold, sign='smaller', run_window=100):
    assert sign in ['smaller', 'bigger'], 'The inequality sign is not correct'
    if key not in data.keys():
        print('The given attribute has not been tracked')
        return np.nan
    run_mean_y = compute_running_mean(data[key], run_window)
    indices = np.where(run_mean_y > threshold)[0] if sign == 'bigger' else np.where(run_mean_y < threshold)[0]
    idx = indices[0] if len(indices) > 0 else np.nan
    return idx

def get_z_factor_experiment(env, kwargs, kwargs_rt, kwargs_op, lrs, seeds, key, threshold, sign, run_window):

    # check if lrs is a list of 3 lists
    assert isinstance(lrs, list), "The object is not a list."
    assert len(lrs) == 3, "The list does not have 3 elements."
    for i, item in enumerate(lrs):
        assert isinstance(item, list), f"The element at index {i} is not a list."

    # check if seeds is a list
    assert isinstance(seeds, list), "The object is not a list."

    def get_info(data, key, threshold, sign, run_window):

        if key not in data.keys():
            return np.nan, np.nan, np.nan, np.nan

        # get the last not nan values of the array data[key]
        not_nan_mask = ~np.isnan(data[key])
        indices = np.where(not_nan_mask)[0]
        last_idx = indices[-1]
        last_key_values = data[key][last_idx-run_window:last_idx].mean()

        # get the number of iterations until the threshold is reached
        n_iter = get_n_iterations_until_goal(data, key, threshold, sign, run_window)

        # get the total number of time steps done at each iteration 
        time_steps = data['max_lengths'][slice(0, n_iter)].sum() if n_iter is not np.nan else np.nan

        # get the computational time at each iteration 
        cts = data['cts'][slice(0, n_iter)].sum() if n_iter is not np.nan else np.nan

        return last_key_values, n_iter, time_steps, cts

    # deterministic policy or stochastic policy
    reinforce_fn = reinforce_stochastic if 'policy_type' in kwargs else reinforce_deterministic

    # preallocate arrays
    lasts = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    n_grad_iters = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    time_steps = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]
    cts = [np.full((len(seeds), len(lrs[i])), np.nan) for i in range(3)]

    for i, seed in enumerate(seeds):

        # random time horizon
        for j, lr in enumerate(lrs[0]):
            succ, data = reinforce_fn(env, lr=lr, seed=seed, **kwargs, **kwargs_rt)
            if succ:
                lasts[0][i, j], n_grad_iters[0][i, j], time_steps[0][i, j], cts[0][i, j] \
                    = get_info(data, key, threshold, sign, run_window)

        # on policy expectation with z-factor estimated 
        for j, lr in enumerate(lrs[1]):
            succ, data = reinforce_fn(env, estimate_z=True, lr=lr, seed=seed, **kwargs, **kwargs_op)
            if succ:
                lasts[1][i, j], n_grad_iters[1][i, j], time_steps[1][i, j], cts[1][i, j] \
                    = get_info(data, key, threshold, sign, run_window)

        # on policy expectation with z-factor neglected 
        for j, lr in enumerate(lrs[2]):
            succ, data = reinforce_fn(env, estimate_z=False, lr=lr, seed=seed, **kwargs, **kwargs_op)
            if succ:
                lasts[2][i, j], n_grad_iters[2][i, j], time_steps[2][i, j], cts[2][i, j] \
                    = get_info(data, key, threshold, sign, run_window)

    return lasts, n_grad_iters, time_steps, cts

