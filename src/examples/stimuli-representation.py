import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
sns.set(style='white', palette='colorblind', context='poster')
np.random.seed(0)

'''how to use'''
# init
n_param, n_branch = 16, 4
pad_len = 0
n_parts = 2
n_samples = 256
p_rm_ob_enc = 0
p_rm_ob_rcl = 0
n_rm_fixed = False
task = SequenceLearning(
    n_param, n_branch, pad_len=pad_len,
    p_rm_ob_enc=p_rm_ob_enc,
    p_rm_ob_rcl=p_rm_ob_rcl,
    n_rm_fixed=n_rm_fixed,
)
# take sample
X, Y = task.sample(n_samples, to_torch=False)
print(f'X shape = {np.shape(X)}, n_example x time x x-dim')
print(f'Y shape = {np.shape(Y)},  n_example x time x y-dim')

'''visualize the sample'''
# pick a sample
i = 0
x, y = X[i], Y[i]
cmap = 'bone'

x_split = np.split(x, (n_param, n_param + n_branch), axis=1)
episodic_sim = False
if episodic_sim:
    x_split[0][:n_param] = np.vstack(
        [x_split[0][:n_param][0], np.eye(n_param, k=-1)[1:, :]]
    )
    x_split[0][n_param:] = np.vstack(
        [x_split[0][n_param:][0], np.eye(n_param, k=-1)[1:, :]]
    )
    for t in range(n_param):
        obs_feature_id_p1 = np.argmax(x_split[0][:n_param][t])
        x_split[1][:n_param][t] = y[obs_feature_id_p1]
        obs_feature_id_p2 = np.argmax(x_split[0][n_param:][t])
        x_split[1][n_param:][t] = y[obs_feature_id_p2]


mat_list = x_split + [y]
f, axes = plt.subplots(
    2, 4, figsize=(12, 9), sharey=True,
    gridspec_kw={
        'width_ratios': [n_param, n_branch, n_param, n_branch],
        'height_ratios': [n_param, n_param]
    },
)
title_list = ['Observed feature', 'Observed value',
              'Queried feature', 'Queried value']
ylabel_list = ['Part one', 'Part two']
for i, mat in enumerate(mat_list):
    [mat_p1, mat_p2] = np.split(mat, [n_param], axis=0)
    axes[0, i].imshow(mat[:n_param, :], cmap=cmap)
    axes[1, i].imshow(mat[n_param:, :], cmap=cmap)
    axes[0, i].set_title(title_list[i], fontname='Helvetica')
    axes[0, i].set_xticks([])

for i in [1, 3]:
    axes[1, i].set_xticks(range(n_branch))
    axes[1, i].set_xticklabels(i for i in np.arange(4) + 1)


for i in range(2):
    axes[i, 0].set_yticks(np.arange(0, n_param, 5))
    axes[i, 0].set_ylabel(ylabel_list[i], fontname='Helvetica')

f.tight_layout()
f.savefig(f'examples/figs/stimulus-rep.png', dpi=100, bbox_inches='tight')
