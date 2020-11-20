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
n_samples = 5
p_rm_ob_enc = 0
p_rm_ob_rcl = 0
n_rm_fixed = False
key_rep_type = 'time'
task = SequenceLearning(
    n_param, n_branch, pad_len=pad_len,
    p_rm_ob_enc=p_rm_ob_enc,
    p_rm_ob_rcl=p_rm_ob_rcl,
    n_rm_fixed=n_rm_fixed,
    key_rep_type=key_rep_type
)
# take sample
X, Y = task.sample(n_samples, to_torch=False)
print(f'X shape = {np.shape(X)}, n_example x time x x-dim')
print(f'Y shape = {np.shape(Y)},  n_example x time x y-dim')

# pick a sample
i = 0
x, y = X[i], Y[i]


'''visualize the sample'''
cmap = 'bone'
f, axes = plt.subplots(
    1, 2, figsize=(12, 10), sharey=True,
    gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]},
)
axes[0].imshow(x, cmap=cmap)
axes[1].imshow(y, cmap=cmap)

axes[0].set_title('Input', fontname='Helvetica')
axes[1].set_title('Target', fontname='Helvetica')
# axes[0].set_xlabel(
#     '\nObserved feature     Observed value    Queried feature    ', fontname='Helvetica')
# axes[1].set_xlabel('\nQueried value', fontname='Helvetica')
axes[0].set_ylabel('Time\n Part two ' + ' ' * 24 +
                   ' Part one', fontname='Helvetica')
# ax.xticks([])

for ax in axes:
    ax.axhline(n_param + pad_len - .5, color='grey', linestyle='--')
    ax.set_xticks([])
axes[0].axvline(task.k_dim - .5, color='red', linestyle='--')
axes[0].axvline(task.k_dim + task.v_dim - .5, color='red', linestyle='--')

f.savefig(f'examples/figs/stimulus-rep.png', dpi=100, bbox_inches='tight')


# '''figure out the dk label'''
#
#
# def get_know_label(t, x, y):
#     know = np.zeros((task.T_total,))
#     # the time points where y_t is presented
#     yt_present = x[:, t] == 1
#     when_yt_present = np.where(yt_present)[0]
#     # y labels at yt_present
#     observed_y = x[yt_present, -task.y_dim:]
#     for tau in range(np.sum(yt_present)):
#         print(observed_y[tau, :])
#         print(np.all(observed_y[tau, :] == y[t]))
#         if np.all(observed_y[tau, :] == y[t]):
#             know[when_yt_present[tau]:] = 1
#     return know
#
#
# # pick a time point
# t = 0
# np.array([get_know_label(t, x, y) for t in range(n_param)]).T

#
# task.pad_len, task.T_part
# for t in range(task.T_total):
#     print(t, np.mod(t, task.T_part), np.mod(t, task.T_part) >= task.pad_len)
