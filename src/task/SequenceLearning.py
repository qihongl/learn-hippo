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
            p_rm_ob_enc=0,
            p_rm_ob_rcl=0,
            n_rm_fixed=True,
            context_dim=1,
            context_drift=False,
            append_context=False,
            key_rep_type='time',
            sampling_mode='enumerative'
    ):
        # build a sampler
        self.stim_sampler = StimSampler(
            n_param, n_branch,
            context_dim=context_dim,
            key_rep_type=key_rep_type,
            n_rm_fixed=n_rm_fixed,
            sampling_mode=sampling_mode,
            context_drift=context_drift
        )
        # graph param
        self.n_param = n_param
        self.n_branch = n_branch
        # "noise" in the obseravtion
        self.p_rm_ob_enc = p_rm_ob_enc
        self.p_rm_ob_rcl = p_rm_ob_rcl
        self.n_rm_fixed = n_rm_fixed
        #
        self.append_context = append_context
        self.context_drift = context_drift
        # task duration
        self.T_part = n_param
        self.n_parts = n_parts
        self.T_total = self.T_part * n_parts
        # task dimension
        self.k_dim = self.stim_sampler.k_dim
        self.v_dim = self.stim_sampler.v_dim
        self.x_dim = self.k_dim + self.v_dim
        self.y_dim = self.v_dim
        if append_context:
            # augment x_dim
            self.c_dim = self.stim_sampler.c_dim
            self.x_dim += self.c_dim

    def sample(self, n_samples, to_torch=False):
        # prealloc
        X = np.zeros((n_samples, self.T_total, self.x_dim))
        Y = np.zeros((n_samples, self.T_total, self.y_dim))
        # generate samples
        for i in range(n_samples):
            sample_i = self.stim_sampler.sample(
                n_parts=self.n_parts,
                p_rm_ob_enc=self.p_rm_ob_enc,
                p_rm_ob_rcl=self.p_rm_ob_rcl,
            )
            X[i], Y[i] = _to_xy(sample_i, self.append_context)
        # formatting
        if to_torch:
            X, Y = to_pth(X), to_pth(Y)
        return X, Y


def _to_xy(sample_, append_context):
    # unpack data
    observations, queries = sample_
    [o_keys_vec, o_vals_vec, o_ctxs_vec] = observations
    [q_keys_vec, q_vals_vec, q_ctxs_vec] = queries
    # form x and y
    x = np.hstack([
        np.vstack([k for k in o_keys_vec]), np.vstack([v for v in o_vals_vec])
    ])
    y = np.vstack(q_vals_vec)
    # if has context
    if append_context:
        x = np.hstack([x, np.vstack([o_ctxs_vec, q_ctxs_vec])])
    return x, y


# '''scratch'''
# # init a graph
# n_param, n_branch = 3, 2
# n_parts = 2
# n_samples = 5
# context_dim = 2
# append_context = True
# sl = SequenceLearning(
#     n_param, n_branch,
#     context_dim=context_dim,
#     append_context=append_context,
# )
# sample = sl.stim_sampler.sample(n_parts)
# # unpack data
# observations, queries = sample
# [o_keys_vec, o_vals_vec, o_ctxs_vec] = observations
# [q_keys_vec, q_vals_vec, q_ctxs_vec] = queries
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
# n_param, n_branch = 6, 2
# n_parts = 2
# n_samples = 5
# context_dim = 10
# append_context = True
# sl = SequenceLearning(
#     n_param, n_branch,
#     context_dim=context_dim,
#     context_drift=False,
#     append_context=append_context,
# )
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
# # assert np.all(x_sorted == np.eye(n_param))
# # assert np.all(y_sorted == y[:T_part])
#
# # plot
# cmap = 'bone'
# f, axes = plt.subplots(
#     1, 2, figsize=(6, 4),
#     gridspec_kw={'width_ratios': [sl.x_dim, sl.y_dim]}
# )
# axes[0].imshow(x, cmap=cmap)
# axes[1].imshow(y, cmap=cmap)
# print(x)
