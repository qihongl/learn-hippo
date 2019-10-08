import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
sns.set(style='white', palette='colorblind', context='talk')
# np.random.seed(2)


n_param, n_branch = 6, 3
n_parts = 3
p_rm_ob_enc = 0.5
p_rm_ob_rcl = 0.5
similarity_cap = .5
# pad_len = 'random'
pad_len = 0
task = SequenceLearning(
    n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl, n_parts=n_parts,
    similarity_cap=similarity_cap
)

n_samples = 10
X, Y, misc = task.sample(
    n_samples, interleave=True, to_torch=False, return_misc=True
)
# get a sample
i = 0
X_ab, Y_ab = X[i], Y[i]
# X_ab, Y_ab = interleave_stories(X, Y, n_parts)
# X_ab, Y_ab = X_ab[0], Y_ab[0]

cmap = 'bone'
f, axes = plt.subplots(
    1, 2, figsize=(6, 12),
    gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]}
)
axes[0].imshow(X_ab, cmap=cmap, vmin=0, vmax=1)
axes[1].imshow(Y_ab, cmap=cmap, vmin=0, vmax=1)

T_total = np.shape(Y_ab)[0]
for eb in np.arange(0, T_total, n_param)[1:]:
    for ax in axes:
        ax.axhline(eb-.5, color='red', linestyle='--')
axes[0].axvline(task.k_dim-.5, color='red', linestyle='--')
axes[0].axvline(task.k_dim+task.v_dim-.5, color='red', linestyle='--')
axes[0].set_xlabel('o-key | o-val | q-key')
axes[1].set_xlabel('q-val')

yticks = [eb-n_param//2 for eb in np.arange(0, T_total+1, n_param)[1:]]
yticklabels = ['A', 'B'] * n_parts
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(yticklabels)

f.savefig(f'examples/figs/seq-learn-ABA.png', dpi=100, bbox_inches='tight')
