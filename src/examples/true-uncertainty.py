'''demo helper functions to compute objective uncertainty '''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
from analysis import batch_compute_true_dk, compute_stats
# from matplotlib.ticker import FormatStrFormatter
sns.set(style='white', palette='colorblind', context='talk')

# build a sampler
n_param = 15
n_branch = 4
pad_len = 3
p_rm_ob_enc = .5
p_rm_ob_rcl = .5
# init
task = SequenceLearning(
    n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
)
# sample
n_samples = 256
X, Y = task.sample(n_samples, to_torch=False)
# unpack
print(f'X shape: {np.shape(X)}, n_samples x T x x_dim')
print(f'Y shape: {np.shape(Y)},  n_samples x T x y_dim')


'''show uncertainty'''

dk_wm, dk_em = batch_compute_true_dk(X, task)
print(f'np.shape(dk_wm): {np.shape(dk_wm)}')
print(f'np.shape(dk_em): {np.shape(dk_em)}')

# compute stats
dk_em_mu, dk_em_er = compute_stats(dk_em)
dk_wm_mu, dk_wm_er = compute_stats(dk_wm)

# plot
f, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.errorbar(
    x=range(len(dk_em_mu)), y=1 - dk_em_mu, yerr=dk_em_er,
    label='w/ EM'
)
ax.errorbar(
    x=np.arange(n_param, n_param * task.n_parts), y=1 - dk_wm_mu, yerr=dk_wm_er,
    label='w/o EM'
)
ax.axvline(n_param, color='grey', linestyle='--')
ax.set_title(f'Expected performance, delay = {pad_len} / {n_param}')
ax.set_xlabel('Time, 0 = prediction onset')
ax.set_ylabel('1 - P(DK)')
ax.legend()
f.tight_layout()
sns.despine()


'''property 1: nested uncertainty structure
As WM info is a subset of EM info, EM DK => WM DK
'''

i = 0
dk_wm_i = dk_wm[i, :]
dk_em_i = dk_em[i, :]
em_useful = np.logical_and(dk_wm_i, ~ dk_em_i[n_param:])
time_em_useful = n_param + np.where(em_useful)[0]

f, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
# axes[0].set_title('As WM info is a subset of EM info, EM DK => WM DK')
axes[0].set_title('Do I know the queried information?')

axes[0].stem(
    np.arange(n_param, n_param * task.n_parts), 1 - dk_wm_i,
    basefmt='grey'
)
axes[1].stem(
    np.arange(n_param * task.n_parts), 1 - dk_em_i,
    basefmt='grey'
)
if len(time_em_useful) > 0:
    axes[1].scatter(
        time_em_useful, np.ones(len(time_em_useful)),
        s=200, facecolors='none', edgecolors='r',
        label=r'$q \in$ EM $\smallsetminus$ WM'
    )
for ax in axes:
    ax.axvline(n_param - .5, color='grey',
               linestyle='--', label='event boundary')
    ax.set_yticks([0, 1])
    ax.set_ylim([-0.05, 1.2])

axes[1].legend()
axes[0].set_ylabel('WM')
axes[1].set_ylabel('EM+WM')
axes[-1].set_xlabel('Time')
sns.despine()
f.tight_layout()


'''property 2: decomposition
EM only, WM only, both, neither
'''

i = 0
dk_wm_i = dk_wm[i, :]
dk_em_i = dk_em[i, :]
em_useful = np.logical_and(dk_wm_i, ~ dk_em_i[n_param:])
time_em_useful = n_param + np.where(em_useful)[0]

f, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
# axes[0].set_title('As WM info is a subset of EM info, EM DK => WM DK')
axes[0].set_title('Do I know the queried information?')

axes[0].stem(
    np.arange(n_param, n_param * task.n_parts), 1 - dk_wm_i,
    basefmt='grey'
)
axes[1].stem(
    np.arange(n_param * task.n_parts), 1 - dk_em_i,
    basefmt='grey'
)
if len(time_em_useful) > 0:
    axes[1].scatter(
        time_em_useful, np.ones(len(time_em_useful)),
        s=200, facecolors='none', edgecolors='r',
        label=r'$q \in$ EM $\smallsetminus$ WM'
    )
for ax in axes:
    ax.axvline(n_param - .5, color='grey',
               linestyle='--', label='event boundary')
    ax.set_yticks([0, 1])
    ax.set_ylim([-0.05, 1.2])

axes[1].legend()
axes[0].set_ylabel('WM')
axes[1].set_ylabel('EM+WM')
axes[-1].set_xlabel('Time')
sns.despine()
f.tight_layout()


'''simulation'''

p_remove_obs = [0, .5]
n_samples = 256

b_pal = sns.color_palette('Blues', n_colors=len(p_remove_obs))
g_pal = sns.color_palette('Greens', n_colors=len(p_remove_obs))

f, ax = plt.subplots(1, 1, figsize=(9, 4))

for ip, p in enumerate(p_remove_obs):
    # sample
    p_rm_ob_enc = p
    p_rm_ob_rcl = p
    task = SequenceLearning(
        n_param=n_param, n_branch=n_branch, pad_len=pad_len,
        p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
    )
    X, Y = task.sample(n_samples, to_torch=False)

    # compute stats
    dk_wm, dk_em = batch_compute_true_dk(X, task)
    dk_em_mu, dk_em_er = compute_stats(dk_em)
    dk_wm_mu, dk_wm_er = compute_stats(dk_wm)

    ax.errorbar(
        x=range(len(dk_em_mu)),
        y=1 - dk_em_mu, yerr=dk_em_er,
        label=f'RM, b={p}', color=b_pal[ip]
    )
    ax.errorbar(
        x=np.arange(n_param, n_param * task.n_parts),
        y=1 - dk_wm_mu, yerr=dk_wm_er,
        label=f'DM, b={p}', color=g_pal[ip]
    )
ax.axvline(n_param, color='grey', linestyle='--')
ax.set_title(f'Expected performance, delay = {pad_len} / {n_param}')
ax.set_xlabel('Time, 0 = prediction onset')
ax.set_ylabel('1 - P(DK)')
ax.set_ylim([-.05, 1.05])
ax.legend(bbox_to_anchor=(1, .8))
f.tight_layout()
sns.despine()
f.savefig(f'examples/figs/baseline-delay{pad_len}.png', dpi=100)
