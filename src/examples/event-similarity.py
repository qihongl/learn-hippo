'''study how different parameters, such as the branching factor of the graph,
affect average event similarity'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task import SequenceLearning
from analysis import compute_event_similarity_matrix
from matplotlib.ticker import FormatStrFormatter
sns.set(style='white', palette='colorblind', context='poster')

'''study inter-event similarity as a function of n_branch, n_param'''
n_param = 15
n_branch = 4
n_samples = 101
expected_similarity = 1 / n_branch

# init
task = SequenceLearning(n_param, n_branch, n_parts=1)
X, Y = task.sample(n_samples, to_torch=False)

# compute similarity
normalize = True
if not normalize:
    similarity_max = task.similarity_max * n_param
else:
    similarity_max = task.similarity_max

similarity_matrix = compute_event_similarity_matrix(Y, normalize=normalize)
# plot the similarity matrix
f, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.heatmap(
    similarity_matrix,
    xticklabels=n_samples // 2, yticklabels=n_samples // 2,
    cmap='viridis', ax=ax
)
ax.set_xlabel('event i')
ax.set_ylabel('event j')
ax.set_title('inter-event similarity')


'''plot the distribution (for the lower triangular part)'''
similarity_matrix_tril = similarity_matrix[np.tril_indices(n_samples, k=-1)]
bins = len(np.unique(similarity_matrix_tril))
linewidth = 10
max_bond = 1 if normalize else n_param
title = 'Inter-event similarity (mu = %.2f, sd= %.2f)' % (
    np.mean(similarity_matrix_tril), np.std(similarity_matrix_tril))
xlabel = '% Param value shared' if normalize else '# Param value shared'
# plot the distribution
f, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.distplot(
    similarity_matrix_tril,
    kde=False, bins=bins, norm_hist=True,
    ax=ax
)
ax.axvline(max_bond, linestyle='--', color='grey', linewidth=linewidth)
ax.axvline(similarity_max, linestyle='--',
           color='grey', alpha=.5, linewidth=linewidth // 2)
ax.set_xlabel(xlabel)
ax.set_ylabel('Freq.')
ax.set_title(title)
ax.set_xlim([0, max_bond])
if normalize:
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
else:
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
sns.despine()


"""
simulate - mean similarity as a function of n_branch
assuming the uniformly distribution next states
"""
n_branch_list = [2, 4, 8, 16]
n_param = 15
n_samples = 100

n = len(n_branch_list)
sim_mu = np.zeros(n,)
sim_sd = np.zeros(n,)
for i, n_branch in enumerate(n_branch_list):
    task = SequenceLearning(n_param, n_branch, n_parts=1)
    X, Y = task.sample(n_samples, to_torch=False)
    similarity_matrix = compute_event_similarity_matrix(Y, normalize=True)
    similarity_matrix_tril = similarity_matrix[np.tril_indices(
        n_samples, k=-1)]
    sim_mu[i] = np.mean(similarity_matrix_tril)
    sim_sd[i] = np.std(similarity_matrix_tril)

f, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.errorbar(x=n_branch_list, y=sim_mu, yerr=sim_sd)
ax.set_title('Inter event similarity ~ graph width')
ax.set_xlabel('Graph width')
ax.set_ylabel('Average % param shared')
ax.set_xticks(n_branch_list)
ax.set_xticklabels(n_branch_list)
f.tight_layout()
sns.despine()


"""
simulate - mean similarity as a function of n_branch
assuming the uniformly distribution next states
"""
n_param_list = [4, 8, 16, 32]
n_branch = 3
n_samples = 100

n = len(n_branch_list)
sim_mu = np.zeros(n,)
sim_sd = np.zeros(n,)
for i, n_param in enumerate(n_param_list):
    task = SequenceLearning(n_param, n_branch, n_parts=1)
    X, Y = task.sample(n_samples, to_torch=False)
    similarity_matrix = compute_event_similarity_matrix(Y, normalize=True)
    similarity_matrix_tril = similarity_matrix[np.tril_indices(
        n_samples, k=-1)]
    sim_mu[i] = np.mean(similarity_matrix_tril)
    sim_sd[i] = np.std(similarity_matrix_tril)

f, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.errorbar(x=n_param_list, y=sim_mu, yerr=sim_sd)
ax.set_title('Inter event similarity ~ graph length')
ax.set_xlabel('Graph length')
ax.set_ylabel('Average % param shared')
ax.set_xticks(n_param_list)
ax.set_xticklabels(n_param_list)
f.tight_layout()
sns.despine()
