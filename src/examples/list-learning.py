import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import ListLearning
sns.set(style='white', palette='colorblind', context='talk')

'''how to use'''
# init a graph
n_param, n_branch = 3, 2
n_parts = 2
n_sample = 10
ll = ListLearning(n_param, n_branch)
X, Y = ll.sample(n_sample)
print(np.shape(X))
print(np.shape(Y))

# show a sample
i = 0
x, y = X[i], Y[i]

cmap = 'bone'
f, axes = plt.subplots(
    1, 2, figsize=(6, 4),
    gridspec_kw={'width_ratios': [ll.x_dim, ll.y_dim]}
)
axes[0].imshow(x, cmap=cmap)
axes[1].imshow(y, cmap=cmap)

axes[0].set_ylabel('Time')
axes[0].set_title('x')
axes[1].set_title('y')
# mark line
axes[0].axvline(ll.k_dim - .5, color='red', linestyle='--')
n_timesteps = n_param
for ax in axes:
    ax.axhline(n_timesteps - .5, color='red', linestyle='--')

f.savefig('examples/figs/list-learning-rnn.png', dpi=100, bbox_inches='tight')
