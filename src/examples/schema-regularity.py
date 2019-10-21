import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
from analysis import compute_event_similarity_matrix, compute_stats
sns.set(style='white', palette='colorblind', context='poster')

'''study inter-event similarity as a function of n_branch, n_param'''
n_param = 10
n_branch = 4
n_samples = 500
# init
# def_path = np.array([np.mod(t, n_branch) for t in np.arange(n_param)])
# def_path = np.ones(n_param,)
def_prob = .5
def_prob = None
task = SequenceLearning(
    n_param, n_branch, n_parts=1,
    # def_path=def_path,
    def_prob=def_prob,
)
# sample
X, Y, misc = task.sample(n_samples, to_torch=False, return_misc=True)
Y_int = np.stack([misc[i][1] for i in range(n_samples)])
_, T_total = np.shape(Y_int)


'''plot next state distribution over time'''

# compute
p_s_next = np.zeros((T_total, n_branch))
for t in range(T_total):
    unique, counts = np.unique(Y_int[:, t], return_counts=True)
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


'''how modify Schematicity impact average event-event similarity'''

n_param = 16
n_branch = 4
def_probs = np.linspace(.25, .9, 5)
n_iter = 5
similarity_max = .75
sim_mu = np.zeros((len(def_probs), n_iter))

for i, def_prob in enumerate(def_probs):
    print(i, def_prob)
    for j in range(n_iter):
        task = SequenceLearning(
            n_param, n_branch, n_parts=1,
            def_prob=def_prob, similarity_max=similarity_max
        )
        X, Y, misc = task.sample(n_samples, to_torch=False, return_misc=True)
        # compute inter-event similarity
        similarity_matrix = compute_event_similarity_matrix(Y, normalize=True)
        similarity_matrix_tril = similarity_matrix[np.tril_indices(
            n_samples, k=-1)]
        sim_mu[i, j] = np.mean(similarity_matrix_tril)


# analysis
mu_sim_mu, er_sim_mu = compute_stats(sim_mu.T)
xticks = range(len(def_probs))
xlabs = ['%.2f' % s for s in def_probs]
n_se = 3

f, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.errorbar(x=xticks, y=mu_sim_mu, yerr=er_sim_mu*n_se)
ax.axhline(1 / n_branch, color='grey', linestyle='--')
ax.set_ylabel('Average event similarity')
ax.set_xlabel('Schematicity (def prob)')
ax.set_xticks(xticks)
ax.set_xticklabels(xlabs)
ax.legend(['lower bound'])
sns.despine()
f.tight_layout()
