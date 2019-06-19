import numpy as np
from utils.utils import to_pth
from task.StimSampler import StimSampler
# import matplotlib.pyplot as plt


class SequenceLearning():
    """a key-value assoc learning task with explicit query keys input,
    - ... where query keys are not explicitly presented
    - but queries are always ordered by "time", so the model knows which
    element is being queried
    """

    def __init__(
            self,
            n_param,
            n_branch,
            n_parts=2,
            key_rep_type='time',
            sampling_mode='enumerative'
    ):
        # build a sampler
        self.stim_sampler = StimSampler(
            n_param, n_branch,
            key_rep_type=key_rep_type,
            sampling_mode=sampling_mode
        )
        # graph param
        self.n_param = n_param
        self.n_branch = n_branch
        # task duration
        self.T_part = n_param
        self.n_parts = n_parts
        self.T_total = self.T_part * n_parts
        # task dimension
        self.k_dim = self.stim_sampler.k_dim
        self.v_dim = self.stim_sampler.v_dim
        self.x_dim = self.k_dim + self.v_dim
        self.y_dim = self.v_dim

    def sample(self, n_samples, to_torch=False):
        # prealloc
        X = np.zeros((n_samples, self.T_total, self.x_dim))
        Y = np.zeros((n_samples, self.T_total, self.y_dim))
        # generate samples
        for i in range(n_samples):
            sample_i = self.stim_sampler.sample(n_parts=self.n_parts)
            X[i], Y[i] = _to_xy(sample_i)
        # formatting
        if to_torch:
            X, Y = to_pth(X), to_pth(Y)
        return X, Y


def _to_xy(sample_):
    [o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec] = sample_
    x = np.hstack([
        np.vstack([k for k in o_keys_vec]),
        np.vstack([v for v in o_vals_vec])
    ])
    y = np.vstack(q_vals_vec)
    return x, y

    # sample_i = self.stim_sampler.sample(n_parts=self.n_parts)
    # [o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec] = sample_i
    # # to RNN form
    # X[i] = np.hstack([
    #     np.vstack([k for k in o_keys_vec]),
    #     np.vstack([v for v in o_vals_vec])
    # ])
    # Y[i] = np.vstack(q_vals_vec)

    #     # to RNN form
    #     X[i], Y[i] = _to_xy(o_keys_vec, o_vals_vec)
    # # formatting
    # if to_torch:
    #     X, Y = to_pth(X), to_pth(Y)
    # return X, Y


# '''scratch'''
# # init a graph
# n_param, n_branch = 3, 2
# n_parts = 2
# n_samples = 5
# sl = SequenceLearning(n_param, n_branch)
# sample = sl.stim_sampler.sample(n_parts)
# # unpack data
# [o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec] = sample
# # [o_keys_vec0, o_keys_vec1] = o_keys_vec
# # [o_vals_vec0, o_vals_vec1] = o_vals_vec
# # [q_keys_vec0, q_keys_vec1] = q_keys_vec
# # [q_vals_vec0, q_vals_vec1] = q_vals_vec
# # to RNN form
# x = np.hstack([
#     np.vstack([k for k in o_keys_vec]),
#     np.vstack([v for v in o_vals_vec])
# ])
# y = np.vstack(q_vals_vec)
#

# '''how to use'''
# n_param, n_branch = 3, 2
# n_parts = 2
# n_samples = 5
# sl = SequenceLearning(n_param, n_branch)
# X, Y = sl.sample(n_samples)
# i = 0
# x, y = X[i], Y[i]
#
# # test
# T_part = n_param
# time_ids = np.argmax(x[:T_part, :sl.k_dim], axis=1)
# sort_ids = np.argsort(time_ids)
# x_sorted = x[:T_part, :sl.k_dim][sort_ids]
# y_sorted = x[:T_part, sl.k_dim:][sort_ids]
# assert np.all(x_sorted == np.eye(n_param))
# assert np.all(y_sorted == y[:T_part])
#
# # plot
# cmap = 'bone'
# f, axes = plt.subplots(
#     1, 2, figsize=(6, 4),
#     gridspec_kw={'width_ratios': [sl.x_dim, sl.y_dim]}
# )
# axes[0].imshow(x, cmap=cmap)
# axes[1].imshow(y, cmap=cmap)
