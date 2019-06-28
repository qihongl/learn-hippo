import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
sns.set(style='white', palette='colorblind', context='talk')
# np.random.seed(2)

'''how to use'''
# init
n_param, n_branch = 3, 2
n_parts = 2
n_samples = 5
p_rm_ob_enc = 0
p_rm_ob_rcl = 0
n_rm_fixed = False
key_rep_type = 'time'
task = SequenceLearning(
    n_param, n_branch,
    append_context=True,
    p_rm_ob_enc=p_rm_ob_enc,
    p_rm_ob_rcl=p_rm_ob_rcl,
    n_rm_fixed=n_rm_fixed,
    key_rep_type=key_rep_type
)
# take sample
X, Y = task.sample(n_samples)

# pick a sample
i = 0
x, y = X[i], Y[i]

'''visualize the sample'''
cmap = 'bone'
f, axes = plt.subplots(
    1, 2, figsize=(8, 6),
    gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]}
)
axes[0].imshow(x, cmap=cmap)
axes[1].imshow(y, cmap=cmap)

axes[0].set_title('x')
axes[1].set_title('y')

axes[0].set_xlabel('key/val')
axes[1].set_xlabel('val')

axes[0].set_ylabel('Time')

n_timesteps = n_param
for ax in axes:
    ax.axhline(n_timesteps-.5, color='red', linestyle='--')
axes[0].axvline(task.k_dim-.5, color='red', linestyle='--')
axes[0].axvline(task.k_dim+task.v_dim-.5, color='red', linestyle='--')

f.savefig(f'examples/figs/seq-learn-rnn-{key_rep_type}.png',
          dpi=100, bbox_inches='tight')


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
