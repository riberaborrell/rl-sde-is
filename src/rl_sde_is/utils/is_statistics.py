import numpy as np

from gym_sde_is.utils.sde import compute_is_functional
from gym_sde_is.utils.statistics import compute_array_statistics, compute_std_and_re

from rl_sde_is.utils.numeric import compute_running_mean
from rl_sde_is.utils.path import load_data, save_data

class ISStatistics(object):

    def __init__(self, eval_freq, eval_batch_size, n_iterations, policy_type='det', iter_str='it.:',
                 track_loss=False, track_is=True, track_l2_error=False, track_ct=False):

        assert policy_type in ['det', 'stoch', 'stoch-mean'], 'Policy type not recognized'
        self.policy_type = policy_type

        # frequency of evaluation and batch size
        self.eval_freq = eval_freq
        self.eval_batch_size = eval_batch_size

        # number of iterations (episodes, grad. iterations or total steps)
        self.n_iterations = n_iterations
        self.n_epochs = self.n_iterations // eval_freq + 1
        self.iter_str = iter_str

        # flags
        self.track_loss = track_loss
        self.track_is = track_is
        self.track_l2_error = track_l2_error
        self.track_ct = track_ct

        # steps
        self.mean_lengths = np.full(self.n_epochs, np.nan)
        self.var_lengths = np.full(self.n_epochs, np.nan)
        self.max_lengths = np.full(self.n_epochs, np.nan)

        # fht
        self.mean_fhts = np.full(self.n_epochs, np.nan)
        self.var_fhts = np.full(self.n_epochs, np.nan)

        # returns
        self.mean_returns = np.full(self.n_epochs, np.nan)
        self.var_returns = np.full(self.n_epochs, np.nan)

        # importance sampling estimator
        if track_is:
            self.mean_I_us = np.full(self.n_epochs, np.nan)
            self.var_I_us = np.full(self.n_epochs, np.nan)
            self.re_I_us = np.full(self.n_epochs, np.nan)

        # losses
        if track_loss:
            self.losses = np.full(self.n_epochs, np.nan)
            self.loss_vars = np.full(self.n_epochs, np.nan)

        # l2 error
        if track_l2_error:
            self.policy_l2_errors = np.full(self.n_epochs, np.nan)

        # computational time
        if track_ct:
            self.cts = np.full(self.n_epochs, np.nan)

    def save_epoch(self, i, env, loss=None, loss_var=None, ct=None):

        if self.track_l2_error:
            assert env.l2_errors is not None, 'L2 error is not provided'
        if self.track_loss:
            assert loss is not None and loss_var is not None, 'Loss is not provided'
        if self.track_ct:
            assert ct is not None, 'CT is not provided'

        self.mean_lengths[i], self.var_lengths[i], _, _ = compute_array_statistics(env.lengths)
        self.max_lengths[i] = np.max(env.lengths)
        self.mean_fhts[i], self.var_fhts[i], _, _ = compute_array_statistics(env.lengths * env.dt)
        self.mean_returns[i], self.var_returns[i], _, _ = compute_array_statistics(env.returns)
        if self.track_is:
            is_functional = compute_is_functional(env.girs_stoch_int,
                                                  env.running_rewards, env.terminal_rewards)
            self.mean_I_us[i], self.var_I_us[i], _, self.re_I_us[i] \
                = compute_array_statistics(is_functional)
        if self.track_loss:
            self.losses[i] = loss
            self.loss_vars[i] = loss_var
        if self.track_l2_error:
            self.policy_l2_errors[i] = np.mean(env.l2_errors)
        if self.track_ct:
            self.cts[i] = ct

    def log_epoch(self, i):
        j = i * self.eval_freq
        msg = self.iter_str + ' {:2d}, '.format(j)
        msg += 'mean return: {:.3e}, var return: {:.1e}, '.format(self.mean_returns[i], self.var_returns[i])
        msg += 'mfht: {:.3e}, '.format(self.mean_fhts[i])
        if self.track_loss:
            msg += 'loss: {:.3e}, '.format(self.losses[i])
        if self.track_is:
            _, re_I_us = compute_std_and_re(self.mean_I_us[i], self.var_I_us[i])
            msg += 'is: mean I^u: {:.3e}, var I^u: {:.1e}, re I^u: {:.1e}, '.format(
                self.mean_I_us[i], self.var_I_us[i], re_I_us
            )
        if self.track_l2_error:
            msg += 'l2 error: {:.3e}, '.format(self.policy_l2_errors[i])
        if self.track_ct:
            msg += 'ct: {:.3e}'.format(self.cts[i])
        print(msg)

    def save_stats(self, dir_path):
        save_data(self.__dict__, dir_path, file_name='eval-{}.npz'.format(self.policy_type))

    def load_stats(self, dir_path):
        # get data dictionary
        data = load_data(dir_path, file_name='eval-{}.npz'.format(self.policy_type))

        assert self.eval_freq == data['eval_freq'], 'eval freq mismatch'
        assert self.eval_batch_size == data['eval_batch_size'], 'eval batch size mismatch'
        assert self.n_iterations == data['n_iterations'], 'iterations mismatch'

        # recover attributes
        for key in data:
            setattr(self, key, data[key])

        # compute missing attributes
        self.std_fhts, self.re_fhts = compute_std_and_re(self.mean_fhts, self.var_fhts)
        self.std_I_us, self.re_I_us = compute_std_and_re(self.mean_I_us, self.var_I_us)

    def get_n_iterations_until_goal(self, attr_name, threshold, sign='smaller', run_window=10):
        assert sign in ['smaller', 'bigger'], 'The inequality sign is not correct'
        if not hasattr(self, attr_name):
            print('The given attribute has not been tracked')
            return np.nan
        run_mean_y = compute_running_mean(getattr(self, attr_name), run_window)
        indices = np.where(run_mean_y > threshold)[0] \
                  if sign == 'bigger' else np.where(run_mean_y < threshold)[0]
        idx = indices[0] * self.eval_freq if len(indices) > 0 else np.nan
        return idx

    def get_stats(self):

        # get number of iterations
        iterations = np.arange(self.n_epochs) * self.eval_freq

        # l2 error
        l2_error = self.policy_l2_errors if hasattr(self, 'policy_l2_errors') else np.nan

        return iterations, self.mean_returns, self.mean_fhts, self.max_lengths, \
               self.mean_I_us, self.re_I_us, l2_error

    def get_stats_multiple_datas(self, datas):

        # get number of iterations
        iterations = np.arange(self.n_epochs) * self.eval_freq

        # preallocate arrays
        array_shape = (len(datas), iterations.shape[0])
        objectives = np.empty(array_shape)
        mfhts = np.empty(array_shape)
        max_lengths = np.empty(array_shape)
        psi_is = np.empty(array_shape)
        re_I_u = np.empty(array_shape)
        l2_error = np.empty(array_shape)

        # load evaluation
        for i, data in enumerate(datas):
            self.load_stats(data['dir_path'])
            objectives[i] = self.mean_returns
            mfhts[i] = self.mean_fhts
            max_lengths[i] = self.max_lengths
            psi_is[i] = self.mean_I_us
            re_I_u[i] = self.re_I_us
            l2_error[i] = self.policy_l2_errors if hasattr(self, 'policy_l2_errors') else np.nan

        return iterations, objectives, mfhts, max_lengths, psi_is, re_I_u, l2_error
