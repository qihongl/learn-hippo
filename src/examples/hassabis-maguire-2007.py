import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='poster', palette='colorblind')

'''hassabis maguire 2007'''
n_groups = 2
group_names = ['patient', 'control']

d_names = 'overall richness'
d_mu = [27.54, 45.06]
d_sd = [13.12, 4.02]
d_n = [5, 10]

'''two bars'''
# cb_pal = sns.color_palette(n_colors=n_groups)
# f, ax = plt.subplots(1, 1, figsize=(4, 4))
# xticks = range(n_groups)
# ax.bar(x=xticks, height=d_mu, yerr=d_sd, color=cb_pal)
# ax.set_xticks(xticks)
# ax.set_xticklabels(group_names)
# ax.set_ylabel('Experiential index')
# ax.set_title('Hassabis & Maguire (2007)\n n patients = %d' % (d_n[0]))
# sns.despine()
# f.tight_layout()
'''patient bar only'''
cb_pal = sns.color_palette(n_colors=n_groups)
f, axes = plt.subplots(
    1, 2, figsize=(12, 7), sharey=True,
    gridspec_kw={'width_ratios': [2, 6]}
)
xticks = range(1)
axes[0].bar(x=xticks, height=d_mu[0]/d_mu[1],
            yerr=d_sd[0]/d_mu[1])
axes[0].axhline(1, color='grey', linestyle='--')

axes[0].set_xlabel(d_names)
axes[0].set_xticklabels([])
axes[0].set_ylabel('Patient performance \nnormalized by controls')
axes[0].set_title('Hassabis & Maguire (2007)\n n patients = %d' % (d_n[0]))
# sns.despine()
# f.tight_layout()


'''keven et al 2018'''

d_names = ['event report \naccuracy',
           'semantic score', 'story-level \nconnectedness']
d_mu = np.array([[13.5, 19.27], [66.5, 65.82], [.75, .91]])
d_sd = np.array([[6.61, 3.58], [15.55, 13.84], [.5, .3]])
d_n = np.array([4, 7])

'''two bars'''
# width = 0.35  # the width of the bars
# x = np.arange(len(d_names))  # the label locations
# f, ax = plt.subplots(1, 1, figsize=(5, 4))
# rects1 = ax.bar(x - width/2, d_mu[:, 0], width, label=group_names[0])
# rects2 = ax.bar(x + width/2, d_mu[:, 1], width, label=group_names[1])
# ax.set_ylabel('Scores')
# ax.set_title('Keven et al. (2018)')
# ax.set_xticks(x)
# ax.set_xticklabels(d_names)
# ax.legend()
# sns.despine()
# f.tight_layout()

'''patient bar only'''
cb_pal = sns.color_palette(n_colors=n_groups)
# f, ax = plt.subplots(1, 1, figsize=(6, 6))
xticks = range(len(d_names))
axes[1].bar(x=xticks, height=d_mu[:, 0]/d_mu[:, 1],
            yerr=d_sd[:, 0]/d_mu[:, 1])
axes[1].axhline(1, color='grey', linestyle='--')
axes[1].set_xticks(xticks)
axes[1].set_xticklabels(d_names, rotation=15)
# axes[1].set_ylabel('Patient performance \nnormalized by controls')
axes[1].set_title('Keven et al. (2018)\n n patients = %d' % (d_n[0]))

y_level = 1.635
fontsize = 22
axes[0].text(
    x=-1, y=y_level, s='A',
    horizontalalignment='center', verticalalignment='center',
    fontsize=fontsize, weight='bold',
    # transform=axes[0].transAxes
)
axes[1].text(
    x=-.7, y=y_level, s='B',
    horizontalalignment='center', verticalalignment='center',
    fontsize=fontsize, weight='bold',
    # transform=axes[1].transAxes
)

sns.despine()
f.tight_layout()
