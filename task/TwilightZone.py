# import torch
import numpy as np
from utils.utils import to_pth
from task.MovieSampler import MovieSampler
# import matplotlib.pyplot as plts


class TwilightZone():

    def __init__(
            self,
            n_param,
            n_branch,
            n_parts=2,
            n_timestep=None,
            p_rm_ob_enc=0,
            p_rm_ob_rcl=0,
            sampling_mode='enumerative'
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        self.n_parts = n_parts
        self.p_rm_ob_enc = p_rm_ob_enc
        self.p_rm_ob_rcl = p_rm_ob_rcl
        # inferred params
        self.x_dim = (n_param * n_branch) * 2 + n_branch
        self.y_dim = n_branch
        if n_timestep is None:
            self.T_part = n_param
        else:
            self.T_part = n_timestep
        self.T_total = self.T_part * n_parts
        # build a sampler
        self.movie_sampler = MovieSampler(
            n_param, n_branch,
            sampling_mode=sampling_mode
        )

    def sample(self, n_samples, to_torch=False, stack=True):
        # prealloc
        if stack:
            X = np.zeros((n_samples, self.T_total, self.x_dim))
            Y = np.zeros((n_samples, self.T_total, self.y_dim))
        else:
            X = np.zeros((n_samples, self.n_parts, self.T_part, self.x_dim))
            Y = np.zeros((n_samples, self.n_parts, self.T_part, self.y_dim))
        # generate samples
        for i in range(n_samples):
            X[i], Y[i] = self.movie_sampler.sample(
                self.T_part,
                n_parts=self.n_parts,
                p_rm_ob_enc=self.p_rm_ob_enc,
                p_rm_ob_rcl=self.p_rm_ob_rcl,
                xy_format=True,
                stack=stack
            )
        # formatting
        if to_torch:
            X, Y = to_pth(X), to_pth(Y)
        return X, Y


# # init a graph
# n_param, n_branch = 3, 2
# n_samples = 5
# p_rm_ob_enc = .5
# tz = TwilightZone(n_param, n_branch, p_rm_ob_enc=p_rm_ob_enc)
# X, Y = tz.sample(n_samples, stack=True)
# print(np.shape(X))
# print(np.shape(Y))

# i = 0
# cmap = 'bone'
# f, axes = plt.subplots(
#     1, 2, figsize=(9, 4), sharey=True,
#     gridspec_kw={'width_ratios': [tz.x_dim, tz.y_dim]}
# )
# axes[0].imshow(X[i], cmap=cmap)
# axes[1].imshow(Y[i], cmap=cmap)
