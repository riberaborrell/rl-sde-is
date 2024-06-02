import numpy as np

from gym_sde_is.utils.logging import compute_array_statistics, compute_std_and_re

from rl_sde_is.utils.path import load_data, save_data

class AISStatistics(object):

    def __init__(self, n_episodes, eval_freq_episodes,
                 eval_batch_size, track_l2_error=False, track_ct=True):

        # frequency of evaluation and number of evaluated episodes
        self.n_episodes = n_episodes
        self.eval_freq_episodes = eval_freq_episodes
        self.eval_batch_size = eval_batch_size
        self.n_epochs = n_episodes // eval_freq_episodes + 1

        # flags
        self.track_l2_error = track_l2_error
        self.track_ct = track_ct

        # steps
        self.mean_length = np.empty(self.n_epochs)
        self.var_length = np.empty(self.n_epochs)

        # fht
        self.mean_fht = np.empty(self.n_epochs)
        self.var_fht = np.empty(self.n_epochs)

        # returns
        self.mean_return = np.empty(self.n_epochs)
        self.var_return = np.empty(self.n_epochs)

        # importance sampling estimator
        self.mean_I_u = np.empty(self.n_epochs)
        self.var_I_u = np.empty(self.n_epochs)

        # l2 error
        if track_l2_error:
            self.policy_l2_error = np.empty(self.n_epochs)

        # computational time
        if track_ct:
            self.ct = np.empty(self.n_epochs)

    def save_epoch(self, i, lengths, fhts, returns, psi_is, policy_l2_error=None, ct=None):
        self.mean_length[i], self.var_length[i], _, _ = compute_array_statistics(lengths)
        self.mean_fht[i], self.var_fht[i], _, _ = compute_array_statistics(fhts)
        self.mean_return[i], self.var_return[i], _, _ = compute_array_statistics(returns)
        self.mean_I_u[i], self.var_I_u[i], _, _ = compute_array_statistics(psi_is)
        if self.track_l2_error and policy_l2_error is not None:
            self.policy_l2_error[i] = np.mean(policy_l2_error)
        if self.track_ct and ct is not None:
            self.ct[i] = ct

    def log_epoch(self, i):
        ep = i * self.eval_freq_episodes
        _, re_I_u = compute_std_and_re(self.mean_I_u[i], self.var_I_u[i])
        msg = 'ep.: {:2d}, '.format(ep)
        msg += 'loss: {:.3e}, var: {:.1e}, '.format(self.mean_return[i], self.var_return[i])
        msg += 'mean I^u: {:.3e}, var I^u: {:.1e}, re I^u: {:.1e}, '.format(
                self.mean_I_u[i], self.var_I_u[i], re_I_u
        )
        msg += 'mfht: {:.3e}'.format(self.mean_fht[i])
        if self.track_l2_error:
            msg += ', l2 error: {:.3e}'.format(self.policy_l2_error[i])
        print(msg)

    def save_stats(self, dir_path):
        save_data(self.__dict__, dir_path, file_name='eval.npz')

    def load_stats(self, dir_path):
        # get data dictionary
        data = load_data(dir_path, file_name='eval.npz')

        # recover attributes
        for key in data:
            setattr(self, key, data[key])

        # compute missing attributes
        self.std_fht, self.re_fht = compute_std_and_re(self.mean_fht, self.var_fht)
        self.std_I_u, self.re_I_u = compute_std_and_re(self.mean_I_u, self.var_I_u)
