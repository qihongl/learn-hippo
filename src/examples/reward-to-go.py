'''the effect of the reward-to-go heuristic'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import to_np
from models import compute_returns
sns.set(style='white', palette='colorblind', context='talk')

# pick some gamma
n_gs = 3
gammas = np.linspace(0, .9, n_gs)
normalize = True

# some example r_t sequences
r = np.array([0, 0, 0, -1, 0, 0, 0, 1])
# r = np.array([0, 0, 1, 0, -2, 0, 0, 1])
# r = np.array([0, 0, 0, 1, 0, 0, 0, 1])
# r = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# r = np.array([1, -1, 1, -1, 1, -1, 1, -1])

# plot
c_pals = sns.color_palette('Blues', n_colors=n_gs)
legend_gammas = ['%.1f' % g for g in gammas]
legend_loc = (1.1, 1)

# visualize R_t and r_t
f, axes = plt.subplots(2, 1, figsize=(8, 7))
for i, gamma in enumerate(gammas):
    returns = to_np(compute_returns(r, gamma=gamma, normalize=normalize))
    print(returns)
    axes[0].plot(returns, color=c_pals[i])
axes[0].plot(r, color='k')
axes[0].axhline(0, color='grey', linestyle='--', alpha=.3)
axes[0].set_xlabel('Time')
axes[0].set_ylabel(r'$R_t$')
axes[0].set_title('Compare return vs. the raw reward')
axes[0].legend(legend_gammas, title='gamma',
               bbox_to_anchor=legend_loc, frameon=False)

# visualize the diffference: R_t vs r_t
c_pals = sns.color_palette('Reds', n_colors=n_gs)
for i, gamma in enumerate(gammas):
    returns = to_np(compute_returns(r, gamma=gamma, normalize=normalize))
    axes[1].plot(returns - r, color=c_pals[i])
axes[1].axhline(0, color='grey', linestyle='--', alpha=.3)
axes[1].set_xlabel('Time')
axes[1].set_ylabel(r'$R_t - r_t$')
axes[1].set_title('Difference from raw reward')
axes[1].legend(legend_gammas, title='gamma',
               bbox_to_anchor=legend_loc, frameon=False)

f.tight_layout()
sns.despine()
