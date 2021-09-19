'''measure the average similarity of event samples'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from task import SequenceLearning
from analysis import compute_event_similarity_matrix, compute_stats
sns.set(style='white', palette='colorblind', context='poster')

'''study inter-event similarity as a function of n_branch, n_param'''
n_param = 16
n_branch = 4
n_samples = 5
similarity_cap_lag = 2

def_prob = .25
similarity_pairs = [[0, .125], [0.0, .4], [0, .9], [.35, .9]]

similarity_labels = ['zero', 'low', 'normal', 'high']
n_conditions = len(similarity_pairs)

event_sims = np.zeros((n_conditions, n_samples - 1))
for i, (similarity_min, similarity_max) in enumerate(similarity_pairs):
    task = SequenceLearning(
        n_param, n_branch, n_parts=2, similarity_cap_lag=similarity_cap_lag,
        similarity_min=similarity_min, similarity_max=similarity_max,
        def_prob=def_prob
    )
    X, Y, Misc = task.sample(n_samples, to_torch=False, return_misc=True)

    # compute inter-event similarity
    similarity_matrix = compute_event_similarity_matrix(Y, normalize=True)
    similarity_matrix_tril = similarity_matrix[np.tril_indices(
        n_samples, k=-1)]

    one_matrix = np.ones((n_samples, n_samples))
    tril_mask = np.tril(one_matrix, k=-1).astype(bool)
    tril_k_mask = np.tril(one_matrix, k=-task.similarity_cap_lag).astype(bool)
    similarity_mask_recent = np.logical_and(tril_mask, ~tril_k_mask)

    event_sims[i] = similarity_matrix[similarity_mask_recent]

'''plot'''
mu, se = compute_stats(event_sims, axis=1)
cps = sns.color_palette(n_colors=len(mu))
f, ax = plt.subplots(1, 1, figsize=(9, 6))
for i in range(n_conditions):
    sns.kdeplot(event_sims[i], ax=ax, label=similarity_labels[i])
ax.legend()
for j, mu_j in enumerate(mu):
    ax.axvline(mu_j, color=cps[j], linestyle='--', alpha=.6)
ax.set_title('Event similarity by condition')
ax.set_xlabel('Event similarity')
sns.despine()
f.tight_layout()


# x = [0, 1]
# x.pop(0)
# x.append(2)
# x
