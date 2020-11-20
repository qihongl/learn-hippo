'''demonstrate how schema regularity works'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
from task.utils import sample_def_tps
from analysis import compute_event_similarity_matrix, compute_stats
sns.set(style='white', palette='colorblind', context='poster')

n_param = 10
n_branch = 4
n_samples = 500
# init
def_path = np.array([np.mod(t, n_branch) for t in np.arange(n_param)])
def_prob = .95
n_def_tps = 5
def_tps = sample_def_tps(n_param, n_def_tps)
task = SequenceLearning(
    n_param, n_branch, n_parts=1,
    # def_path=def_path,
    def_prob=def_prob,
    def_tps=def_tps,
)
# sample
X, Y, misc = task.sample(n_samples, to_torch=False, return_misc=True)
Y_int = np.stack([misc[i][1] for i in range(n_samples)])
_, T_total = np.shape(Y_int)

'''plot next state distribution over time'''
p_s_next = np.zeros((T_total, n_branch))
for t in range(T_total):
    unique, counts = np.unique(Y_int[:, t], return_counts=True)
    p_s_next[t, :] = counts / n_samples

# plot
f, ax = plt.subplots(1, 1, figsize=(8, 12))
sns.heatmap(
    p_s_next,
    vmin=0, vmax=1, annot=True,
    cmap='Blues', ax=ax
)
ax.set_title('P(s next = i)')
ax.set_ylabel('Time')
ax.set_xlabel('Next state')
f.tight_layout()
