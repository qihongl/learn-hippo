'''for the schema regularity simulation
- what does it mean to have high-low schema regularity '''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from task import SequenceLearning
# from analysis import compute_event_similarity_matrix
# from matplotlib.ticker import FormatStrFormatter
sns.set(style='white', palette='colorblind', context='poster')

n_branch = 4
xrange = np.arange(n_branch)
high_schema_bar = [.1, .7, .1, .1]
low_schema_bar = [.23, .31, .23, .23]


f, axes = plt.subplots(1, 2, figsize=(7, 6), sharey=True)
axes[0].bar(x=xrange, height=high_schema_bar)
axes[1].bar(x=xrange, height=low_schema_bar)
axes[0].set_title('High')
axes[1].set_title('Low')
f.suptitle('Schema level')
for ax in axes:
    ax.axhline(1 / n_branch, color='grey', linestyle='--', label='chance')
    ax.set_yticklabels([])
    ax.set_xlabel('Upcoming \nstates')
    ax.set_ylabel('Probability')
    ax.set_xticks(xrange)
    ax.set_xticklabels(xrange + 1)

axes[1].legend()
f.tight_layout()
sns.despine()
