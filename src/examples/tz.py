import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task import TwilightZone
sns.set(style='white', palette='colorblind', context='talk')

# init a graph
n_param, n_branch = 3, 2
n_samples = 5
p_rm_ob_enc = .5
tz = TwilightZone(n_param, n_branch, p_rm_ob_enc=p_rm_ob_enc)

'''stack the n_parts'''
X, Y = tz.sample(n_samples, stack=False)
print('stack=False')
print(np.shape(X))
print(np.shape(Y))

X, Y = tz.sample(n_samples, stack=True)
print('stack=True')
print(np.shape(X))
print(np.shape(Y))

'''show an example'''
i = 0
cmap = 'bone'
f, axes = plt.subplots(
    1, 2, figsize=(9, 4), sharey=True,
    gridspec_kw={'width_ratios': [tz.x_dim, tz.y_dim]}
)
axes[0].imshow(X[i], cmap=cmap)
axes[1].imshow(Y[i], cmap=cmap)
axes[0].set_ylabel('time')
axes[0].set_xlabel('x dim')
axes[1].set_xlabel('y dim')
axes[0].set_title('X')
axes[1].set_title('Y')
# f.savefig('examples/figs/tz-rnn.png', dpi=100, bbox_inches='tight')
