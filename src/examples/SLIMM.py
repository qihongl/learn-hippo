import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='talk')

heights = [4, 2, 4]
xlabels = ['Congruent', 'Control', 'Incongruent']
xticks = range(len(xlabels))


f, ax = plt.subplots(1, 1, figsize=(6, 5))
cpal = sns.color_palette(n_colors=len(xlabels))
cpal = [cpal[0], cpal[2], cpal[1]]
ax.bar(x=xticks, height=heights, color=cpal)
ax.set_xticks(xticks)
ax.set_yticks([])
ax.set_ylabel('Memory')
ax.set_xlabel('Condition')
ax.set_xticklabels(xlabels)
sns.despine()
f.tight_layout()
