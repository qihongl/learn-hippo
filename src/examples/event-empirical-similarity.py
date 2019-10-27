import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from task import SequenceLearning
from analysis import compute_event_similarity_matrix, compute_stats
# from matplotlib.ticker import FormatStrFormatter
sns.set(style='white', palette='colorblind', context='poster')

'''study inter-event similarity as a function of n_branch, n_param'''
n_param = 16
n_branch = 4
n_samples = 300
similarity_cap_lag = 2

def_probs = [.25, .9]
def_prob = .25
# similarity_pairs = [[0.0, 1], [.35, .75]]
similarity_pairs = [[0.0, .35], [.35, .7]]

similarity_labels = ['low', 'high']
n_conditions = len(similarity_pairs)

event_sims = np.zeros((n_conditions, n_samples-1))
# for i, def_prob in enumerate(def_probs):
#     similarity_min, similarity_max = [0, .9]
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

'''analysis'''
similarity_vals_df = np.ravel(event_sims)
similarity_labels_df = list(itertools.chain(
    *[[sl] * (n_samples-1) for sl in similarity_labels])
)
col_names = ['similarity', 'condition']
data = [similarity_vals_df, similarity_labels_df]
data_dict = dict(zip(col_names, data))

df = pd.DataFrame.from_dict(data_dict)
f, ax = plt.subplots(1, 1, figsize=(6, 5))
# sns.barplot(
#     x=col_names[1], y=col_names[0], ci=99,
#     data=df, ax=ax
# )
sns.boxplot(
    x=col_names[1], y=col_names[0],
    data=df, ax=ax
)
# ax.set_ylim([0, .6])
ax.set_ylabel('Event similarity')
ax.set_xlabel('Condition')
sns.despine()
f.tight_layout()
