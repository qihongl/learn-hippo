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
p_rm_ob_enc = 1/3
key_rep_type = 'time'
sl = SequenceLearning(
    n_param, n_branch,
    p_rm_ob_enc=p_rm_ob_enc,
    key_rep_type=key_rep_type
)
# take sample
X, Y = sl.sample(n_samples)

# plot a sample
i = 0
x, y = X[i], Y[i]

# plot
cmap = 'bone'
f, axes = plt.subplots(
    1, 2, figsize=(8, 6),
    gridspec_kw={'width_ratios': [sl.x_dim, sl.y_dim]}
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
axes[0].axvline(sl.k_dim-.5, color='red', linestyle='--')

f.savefig(f'examples/figs/seq-learn-rnn-{key_rep_type}.png',
          dpi=100, bbox_inches='tight')
