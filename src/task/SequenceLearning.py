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
            pad_len=0,
            n_parts=2,
            p_rm_ob_enc=0,
            p_rm_ob_rcl=0,
            n_rm_fixed=False,
            permute_queries=False,
            key_rep_type='time',
            sampling_mode='enumerative'
    ):
        # build a sampler
        self.stim_sampler = StimSampler(
            n_param, n_branch,
            pad_len=pad_len,
            key_rep_type=key_rep_type,
            n_rm_fixed=n_rm_fixed,
            sampling_mode=sampling_mode,
        )
        # graph param
        self.n_param = n_param
        self.n_branch = n_branch
        self.pad_len = pad_len
        # "noise" in the obseravtion
        self.p_rm_ob_enc = p_rm_ob_enc
        self.p_rm_ob_rcl = p_rm_ob_rcl
        self.n_rm_fixed = n_rm_fixed
        # whether to permute queries
        self.permute_queries = permute_queries
        # task duration
        self.T_part = n_param + pad_len
        self.n_parts = n_parts
        self.T_total = self.T_part * n_parts
        # task dimension
        self.k_dim = self.stim_sampler.k_dim
        self.v_dim = self.stim_sampler.v_dim
        self.x_dim = self.k_dim * 2 + self.v_dim
        self.y_dim = self.v_dim

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
                permute_queries=self.permute_queries,
            )
            X[i], Y[i] = _to_xy(sample_i)
        # formatting
        if to_torch:
            X, Y = to_pth(X), to_pth(Y)
        return X, Y


def _to_xy(sample_):
    # unpack data
    observations, queries = sample_
    [o_keys_vec, o_vals_vec, o_ctxs_vec] = observations
    [q_keys_vec, q_vals_vec, q_ctxs_vec] = queries
    # form x and y
    x = np.hstack([
        np.vstack([k for k in o_keys_vec]),
        np.vstack([v for v in o_vals_vec]),
        np.vstack([k for k in q_keys_vec]),
    ])
    y = np.vstack(q_vals_vec)
    return x, y


# '''scratch'''
# # init a graph
# n_param, n_branch = 6, 2
# n_parts = 2
# n_samples = 5
# append_context = True
# task = SequenceLearning(
#     n_param, n_branch,
#     # context_onehot=True,
#     # context_drift=True,
#     # append_context=append_context,
# )
# sample = task.stim_sampler.sample(n_parts)
# # unpack data
# observations, queries = sample
# [o_keys_vec, o_vals_vec, o_ctxs_vec] = observations
# [q_keys_vec, q_vals_vec, q_ctxs_vec] = queries
# # to RNN form
# x = np.hstack([
#     np.vstack([k for k in o_keys_vec]),
#     np.vstack([v for v in o_vals_vec])
# ])
# y = np.vstack(q_vals_vec)


'''how to use'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_param, n_branch = 6, 2
    n_parts = 2
    n_samples = 5
    pad_len = 3
    permute_queries = False
    task = SequenceLearning(
        n_param, n_branch,
        pad_len=pad_len,
        permute_queries=permute_queries
    )

    # gen samples
    X, Y = task.sample(n_samples)
    i = 0
    x, y = X[i], Y[i]

    # plot
    cmap = 'bone'
    f, axes = plt.subplots(
        1, 2, figsize=(6, 6),
        gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]}
    )
    axes[0].imshow(x, cmap=cmap, vmin=0, vmax=1)
    axes[1].imshow(y, cmap=cmap, vmin=0, vmax=1)
    print(x)

    for ax in axes:
        ax.axhline(task.T_part-.5, color='red', linestyle='--')
    axes[0].axvline(task.k_dim-.5, color='red', linestyle='--')
    axes[0].axvline(task.k_dim+task.v_dim-.5, color='red', linestyle='--')
