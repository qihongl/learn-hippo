import numpy as np
from utils.utils import to_pth
from task.StimSampler import StimSampler
# import matplotlib.pyplot as plt


class ListLearning():

    def __init__(
            self,
            n_param,
            n_branch,
            n_parts=2,
            n_timestep=None,
            sampling_mode='enumerative'
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        self.n_parts = n_parts
        # inferred params
        self.x_dim = n_param * n_branch + n_branch
        self.y_dim = n_branch
        if n_timestep is None:
            self.T_part = n_param
        else:
            self.T_part = n_timestep
        self.T_total = self.T_part * n_parts
        # build a sampler
        self.stim_sampler = StimSampler(
            n_param, n_branch,
            sampling_mode=sampling_mode
        )

    def sample(
            self, n_samples,
            to_torch=False
    ):
        # prealloc
        X = np.zeros((n_samples, self.T_total, self.x_dim))
        Y = np.zeros((n_samples, self.T_total, self.y_dim))
        # generate samples
        for i in range(n_samples):
            [keys_vec, vals_vec], _ = self.stim_sampler.sample(
                self.T_part,
                n_parts=self.n_parts,
                xy_format=False,
            )
            X[i], Y[i] = _to_xy(keys_vec, vals_vec)
        # formatting
        if to_torch:
            X, Y = to_pth(X), to_pth(Y)
        return X, Y


def _to_xy(keys_vec, vals_vec):
    """construct x and y

    Parameters
    ----------
    keys_vec : type
        Description of parameter `keys_vec`.
    vals_vec : type
        Description of parameter `vals_vec`.

    Returns
    -------
    type
        Description of returned object.

    """
    [o_keys_vec, q_keys_vec] = keys_vec
    [o_vals_vec, q_vals_vec] = vals_vec
    x = np.vstack([
        np.hstack([o_keys_vec, o_vals_vec]),
        np.hstack([q_keys_vec, np.zeros_like(q_vals_vec)])
    ])
    y = np.vstack([
        o_vals_vec,
        q_vals_vec
    ])
    return x, y


# '''how to use'''
# # init a graph
# n_param, n_branch = 3, 2
# n_timesteps = n_param
# n_parts = 2
# ll = ListLearning(n_param, n_branch)
# # show a sample
# n_sample = 2
# X, Y = ll.sample(n_sample)
# i = 0
# x, y = X[i], Y[i]
# f, axes = plt.subplots(
#     1, 2, figsize=(6, 4),
#     gridspec_kw={'width_ratios': [n_param * n_branch+n_branch, n_branch]}
# )
# axes[0].imshow(x)
# axes[1].imshow(y)

# '''raw'''
# sampler = StimSampler(n_param, n_branch)
# sample_ = sampler.sample(
#     n_timesteps, n_parts, p_rm_ob_enc, p_rm_ob_rcl,
#     xy_format=False, stack=True
# )
# [keys_vec, vals_vec], _ = sample_
# [o_keys_vec, q_keys_vec] = keys_vec
# [o_vals_vec, q_vals_vec] = vals_vec
#
# cmap = 'bone'
# f, axes = plt.subplots(
#     2, 2, figsize=(9, 7), sharey=True,
#     gridspec_kw={'width_ratios': [n_param * n_branch, n_branch]}
# )
# axes[0, 0].imshow(o_keys_vec, cmap=cmap)
# axes[1, 0].imshow(q_keys_vec, cmap=cmap)
# axes[0, 1].imshow(o_vals_vec, cmap=cmap)
# axes[1, 1].imshow(q_vals_vec, cmap=cmap)
#
# # label
# axes[-1, 0].set_xlabel('Keys/States')
# axes[-1, 1].set_xlabel('Values/Action')
# axes[0, 0].set_ylabel(f'Time, o phase')
# axes[1, 0].set_ylabel(f'Time, q phase')
#
# for ip in range(n_parts):
#     axes[ip, 0].set_yticks(range(n_timesteps))
#     axes[ip, 0].set_yticklabels(range(n_timesteps))
# f.tight_layout()
