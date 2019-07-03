import numpy as np
from utils.utils import to_pth
from task.utils import get_event_ends
from task.StimSampler import StimSampler


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
            max_pad_len=None,
            n_parts=2,
            def_path=None,
            def_prob=None,
            p_rm_ob_enc=0,
            p_rm_ob_rcl=0,
            n_rm_fixed=False,
            permute_queries=False,
            key_rep_type='time',
            sampling_mode='enumerative'
    ):
        # build a sampler
        self.stim_sampler = StimSampler(
            n_param=n_param,
            n_branch=n_branch,
            pad_len=pad_len,
            max_pad_len=max_pad_len,
            def_path=def_path,
            def_prob=def_prob,
            key_rep_type=key_rep_type,
            n_rm_fixed=n_rm_fixed,
            sampling_mode=sampling_mode,
        )
        # graph param
        self.n_param = n_param
        self.n_branch = n_branch
        self.n_parts = n_parts
        self.pad_len = pad_len
        # #
        self.max_pad_len = self.stim_sampler.max_pad_len
        self.T_part_max = self.n_param + self.max_pad_len
        self.T_total_max = self.T_part_max * self.n_parts
        # #
        # self.T_part_min = self.n_param
        # self.T_total_min = self.T_part_min * self.n_parts
        # "noise" in the obseravtion
        self.p_rm_ob_enc = p_rm_ob_enc
        self.p_rm_ob_rcl = p_rm_ob_rcl
        self.n_rm_fixed = n_rm_fixed
        # whether to permute queries
        self.permute_queries = permute_queries
        # task dimension
        self.k_dim = self.stim_sampler.k_dim
        self.v_dim = self.stim_sampler.v_dim
        self.x_dim = self.k_dim * 2 + self.v_dim
        self.y_dim = self.v_dim

    def sample(self, n_samples, to_torch=True):
        # prealloc, agnostic about sequence length
        X = [None] * n_samples
        Y = [None] * n_samples
        # generate samples
        for i in range(n_samples):
            sample_i = self.stim_sampler.sample(
                n_parts=self.n_parts,
                p_rm_ob_enc=self.p_rm_ob_enc,
                p_rm_ob_rcl=self.p_rm_ob_rcl,
                permute_queries=self.permute_queries,
            )
            # convert to rnn form
            X[i], Y[i] = _to_xy(sample_i)
        # type conversion
        if to_torch:
            X = [to_pth(X[i]) for i in range(n_samples)]
            Y = [to_pth(Y[i]) for i in range(n_samples)]
        return X, Y

    def get_time_param(self, T_total):
        """compute time related parameters
        since it might be unique for each example

        Parameters
        ----------
        T_total : int
            the 0-th dim of X_i or Y_i, i.e. the input sequnce length
            T_total = (n_param + pad_len) x n_parts

        Returns
        -------
        type
            Description of returned object.

        """
        T_part = T_total // self.n_parts
        pad_len = T_part - self.n_param
        event_ends = get_event_ends(T_part, self.n_parts)
        event_bond = event_ends[0]+1
        return T_part, pad_len, event_ends, event_bond


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
    pad_len = 'random'
    task = SequenceLearning(
        n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    )

    # gen samples
    X, Y = task.sample(n_samples)

    # get a sample
    i = 0
    X_i, Y_i = X[i], Y[i]

    # compute time info
    T_total = np.shape(X_i)[0]
    T_part, pad_len, event_ends, event_bond = task.get_time_param(T_total)

    # plot
    cmap = 'bone'
    f, axes = plt.subplots(
        1, 2, figsize=(6, 6),
        gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]}
    )
    axes[0].imshow(X_i, cmap=cmap, vmin=0, vmax=1)
    axes[1].imshow(Y_i, cmap=cmap, vmin=0, vmax=1)
    # print(task.event_ends)
    # print(task.event_bond)

    for ax in axes:
        ax.axhline(event_bond-.5, color='red', linestyle='--')
    axes[0].axvline(task.k_dim-.5, color='red', linestyle='--')
    axes[0].axvline(task.k_dim+task.v_dim-.5, color='red', linestyle='--')
