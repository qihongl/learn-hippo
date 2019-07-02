import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
sns.set(style='white', palette='colorblind', context='poster')

'''study inter-event similarity as a function of n_branch, n_param'''
n_param = 10
n_branch = 4
n_samples = 500
# init
def_path = np.ones(n_param,)
def_path = np.array([np.mod(t, n_branch) for t in np.arange(n_param)])
def_prob = .5
task = SequenceLearning(
    n_param, n_branch, n_parts=1,
    # def_path=def_path,
    # def_prob=def_prob,
)
# sample
X, Y = task.sample(n_samples, to_torch=False)
# unpack
print(np.shape(X))
print(np.shape(Y))

'''plot next state distribution over time'''

# compute
p_s_next = np.zeros((task.T_total, n_branch))
for t in range(task.T_total):
    Y_feature_vector = np.argmax(Y, axis=2)
    unique, counts = np.unique(Y_feature_vector[:, t], return_counts=True)
    p_s_next[t, :] = counts / n_samples

# plot
f, ax = plt.subplots(1, 1, figsize=(6, 10))
sns.heatmap(
    p_s_next,
    vmin=0, vmax=1, annot=True,
    cmap='Blues', ax=ax
)
ax.set_title('P(s next = i)')
ax.set_ylabel('Time')
ax.set_xlabel('Next state')
# f.tight_layout
