'''demo: memory benefit
1. even if there is no negative effect of recall (lure, noise on target, etc.)
memory benefit still decrease sup-linearly
2. how does delay affect memory benefit
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
sns.set(style='white', context='poster')


def compute_baseline(t, T, chance):
    return chance * (T - t) / T + t / T


def compute_fills(tao):
    x_fill = np.arange(tao, T, .01)
    baseline_fill = np.linspace(baseline[tao], baseline[-1], num=len(x_fill))
    ceil_fill = np.ones(len(x_fill),)
    return x_fill, baseline_fill, ceil_fill


T = 30
branching_factor = 3
chance = 1 / branching_factor
baseline = [compute_baseline(t, T, chance)for t in np.arange(0, T + 1, 1)]


# compute normalized memory benefit
memory_benefit = np.zeros(T)
memory_benefit_normalized = np.zeros(T)
total_memory_benefit = (1 - chance) * T / 2
for tao_ in range(T):
    height = 1 - baseline[tao_]
    base = T - tao_
    memory_benefit[tao_] = height * base / 2
    memory_benefit_normalized[tao_] = memory_benefit[tao_] / \
        total_memory_benefit

cum_memory_benefit = np.zeros_like(memory_benefit)
for i in range(len(memory_benefit)):
    cum_memory_benefit[i] = np.sum(memory_benefit[i:])

# decide recall timing
tao = 5
x_fill, baseline_fill, ceil_fill = compute_fills(tao)

# plot
alpha = .5
# with plt.xkcd():
# plt.rc('legend', **{'fontsize': 20})
text_x = -.12
text_y = .97

f, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].plot(
    baseline, color='black', linestyle='--', linewidth=4, label='baseline'
)
axes[0].axvline(tao, linestyle='--')
axes[0].fill_between(
    x_fill, baseline_fill, ceil_fill,
    alpha=alpha, label='memory benefit'
)
axes[0].axhline(1, linestyle='--', color='grey')
axes[0].axhline(chance, linestyle='--', color='grey')
axes[0].set_ylabel('Prediction accuracy')
axes[0].set_yticks([])
axes[0].set_xticks([tao])
axes[0].set_xticklabels(['recall time'])
axes[0].legend(bbox_to_anchor=(.5, .45), frameon=False)
axes[0].text(text_x, text_y, 'A',
             ha='center', va='center', transform=axes[0].transAxes,
             fontsize=28, fontweight='bold')

min_darkness = 1
pad_lengths = [1, 6, 12]
pad_lengths_names = ['no delay', 'short delay', 'long delay']
b_pal = sns.color_palette("Blues", n_colors=len(pad_lengths) + min_darkness)
b_pal = b_pal[min_darkness:]
lgd = [Line2D([0], [0], color=b_pal[i], label=pad_lengths_names[i])
       for i in range(len(pad_lengths))]
for i_pl, pd_len in enumerate(pad_lengths):
    axes[1].plot(cum_memory_benefit[pd_len - 1:],
                 color=b_pal[i_pl], linewidth=4)
leg_ = axes[1].legend(
    title='prediction delay time', handles=lgd, frameon=False,
    bbox_to_anchor=(.55, .3)
)
leg_.set_title("prediction delay", prop={'size': 'medium'})
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Memory benefit')
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].text(text_x, text_y, 'B',
             ha='center', va='center', transform=axes[1].transAxes,
             fontsize=28, fontweight='bold')
f.tight_layout()
sns.despine()
f.savefig('examples/figs/demo_mb.png', bbox_inches='tight', dpi=100)
