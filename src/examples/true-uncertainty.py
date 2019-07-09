'''demo helper functions to compute objective uncertainty '''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
from analysis import batch_compute_true_dk, compute_stats
sns.set(style='white', palette='colorblind', context='poster')

# build a sampler
n_param = 15
n_branch = 4
pad_len = 0
p_rm_ob_enc = 1
p_rm_ob_rcl = 1
# init
task = SequenceLearning(
    n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
)
# sample
n_samples = 200
X, Y = task.sample(n_samples, to_torch=False)
# unpack
print(f'X shape: {np.shape(X)}, n_samples x T x x_dim')
print(f'Y shape: {np.shape(Y)},  n_samples x T x y_dim')

# plt.imshow(X[0])
# plt.imshow(Y[0])

# compute stats
dk_wm, dk_em = batch_compute_true_dk(X, task)
dk_em_mu, dk_em_er = compute_stats(dk_em)
dk_wm_mu, dk_wm_er = compute_stats(dk_wm)

f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.errorbar(
    x=range(len(dk_em_mu)), y=dk_em_mu, yerr=dk_em_er,
    label='w/  EM'
)
ax.errorbar(
    x=np.arange(n_param, n_param * task.n_parts), y=dk_wm_mu, yerr=dk_wm_er,
    label='w/o EM'
)
ax.axvline(n_param, color='grey', linestyle='--')
ax.set_title(f'Objective uncertainty, delay = {pad_len} / {n_param}')
ax.set_xlabel('Time, 0 = prediction onset')
ax.set_ylabel('P(DK)')
ax.legend()
f.tight_layout()
sns.despine()
