import numpy as np

from gym_sde_is.utils.logging import compute_array_statistics, compute_std_and_re

from rl_sde_is.utils.path import load_data, save_data

class ISStatistics(object):

    def __init__(self, eval_freq, eval_batch_size, track_loss=False, track_is=True,
                 track_l2_error=False, track_ct=True, **kwargs):

        # frequency of evaluation and batch size
        self.eval_freq = eval_freq
        self.eval_batch_size = eval_batch_size

        # number of episodes, iterations or total steps
        if 'n_episodes' in kwargs.keys():
            self.n_episodes = kwargs['n_episodes']
            self.n_epochs = self.n_episodes // eval_freq + 1
            self.iter_str = 'ep. :'
        elif 'n_iterations' in kwargs.keys():
            self.n_iterations = kwargs['n_iterations']
            self.n_epochs = self.n_iterations // eval_freq + 1
            self.iter_str = 'it. :'
        elif 'n_total_steps' in kwargs.keys():
            self.n_total_steps = kwargs['n_total_steps']
            self.n_epochs = self.n_total_steps // eval_freq + 1
            self.iter_str = 'n:'

        # flags
        self.track_loss = track_loss
        self.track_is = track_is
        self.track_l2_error = track_l2_error
        self.track_ct = track_ct

        # steps
        self.mean_lengths = np.empty(self.n_epochs)
        self.var_lengths = np.empty(self.n_epochs)

        # fht
        self.mean_fhts = np.empty(self.n_epochs)
        self.var_fhts = np.empty(self.n_epochs)

        # returns
        self.mean_returns = np.empty(self.n_epochs)
        self.var_returns = np.empty(self.n_epochs)

        # importance sampling estimator
        if track_is:
            self.mean_I_us = np.empty(self.n_epochs)
            self.var_I_us = np.empty(self.n_epochs)

        # losses
        if track_loss:
            self.losses = np.empty(self.n_epochs)

        # l2 error
        if track_l2_error:
            self.policy_l2_errors = np.empty(self.n_epochs)

        # computational time
        if track_ct:
            self.cts = np.empty(self.n_epochs)

    def save_epoch(self, i, lengths, fhts, returns, psi_is=None,
                   policy_l2_errors=None, loss=None, ct=None):
        self.mean_lengths[i], self.var_lengths[i], _, _ = compute_array_statistics(lengths)
        self.mean_fhts[i], self.var_fhts[i], _, _ = compute_array_statistics(fhts)
        self.mean_returns[i], self.var_returns[i], _, _ = compute_array_statistics(returns)
        if self.track_is:
            self.mean_I_us[i], self.var_I_us[i], _, _ = compute_array_statistics(psi_is)
        if self.track_loss and loss is not None:
            self.losses[i] = loss
        if self.track_l2_error and policy_l2_errors is not None:
            self.policy_l2_errors[i] = np.mean(policy_l2_errors)
        if self.track_ct and ct is not None:
            self.cts[i] = ct

    def log_epoch(self, i):
        j = i * self.eval_freq
        _, re_I_us = compute_std_and_re(self.mean_I_us[i], self.var_I_us[i])
        msg = self.iter_str + ': {:2d}, '.format(j)
        msg += 'mean return: {:.3e}, var return: {:.1e}, '.format(self.mean_returns[i], self.var_returns[i])
        msg += 'mfht: {:.3e}, '.format(self.mean_fhts[i])
        if self.track_loss:
            msg += 'loss: {:.3e}, '.format(self.losses[i])
        if self.track_is:
            msg += 'is: mean I^u: {:.3e}, var I^u: {:.1e}, re I^u: {:.1e}, '.format(
                self.mean_I_us[i], self.var_I_us[i], re_I_us
            )
        if self.track_l2_error:
            msg += 'l2 error: {:.3e}'.format(self.policy_l2_errors[i])
        print(msg)

    def save_stats(self, dir_path):
        save_data(self.__dict__, dir_path, file_name='eval.npz')

    def load_stats(self, dir_path):
        # get data dictionary
        data = load_data(dir_path, file_name='eval.npz')

        assert self.eval_freq == data['eval_freq'], 'eval freq mismatch'
        assert self.eval_batch_size == data['eval_batch_size'], 'eval batch size mismatch'

        if hasattr(self, 'n_episodes'):
            assert self.n_episodes == data['n_episodes'], 'n_episodes mismatch'
        elif hasattr(self, 'n_iterations'):
            assert self.n_iterations == data['n_iterations'], 'n_iterations mismatch'
        elif hasattr(self, 'n_total_steps'):
            assert self.n_total_steps == data['n_total_steps'], 'n_total_steps mismatch'

        # recover attributes
        for key in data:
            setattr(self, key, data[key])

        # compute missing attributes
        self.std_fhts, self.re_fhts = compute_std_and_re(self.mean_fhts, self.var_fhts)
        self.std_I_us, self.re_I_us = compute_std_and_re(self.mean_I_us, self.var_I_us)
