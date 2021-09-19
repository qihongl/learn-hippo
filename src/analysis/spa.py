import numpy as np
from scipy.stats import sem
from copy import deepcopy
from models.EM import compute_similarities, transform_similarities
from utils.utils import to_np, to_pth

# def compute_cell_memory_similarity(
#         C, V, inpt, comp, leak=0,
#         kernel='cosine', recall_func='LCA'
# ):

#     n_examples, n_timepoints, n_dim = np.shape(C)
#     n_memories = len(V[0][0])
#     # prealloc
#     sim_raw = np.zeros((n_examples, n_timepoints, n_memories))
#     sim_lca = np.zeros((n_examples, n_timepoints, n_memories))
#     for i in range(n_examples):
#         # compute similarity
#         for t in range(n_timepoints):
#             # compute raw similarity
#             sim_raw[i, t, :] = to_np(compute_similarities(
#                 to_pth(C[i, t]), V[i][0], kernel))
#             # compute LCA similarity
#             sim_lca[i, t, :] = transform_similarities(
#                 to_pth(sim_raw[i, t, :]), recall_func,
#                 leak=leak, comp=comp, w_input=inpt[i, t]
#             )
#     return sim_raw, sim_lca


def compute_cell_memory_similarity(
        C, V, em_gate, cmpt,
):
    n_examples = np.shape(C)[0]
    sim_raw, sim_lca = [None] * n_examples, [None] * n_examples
    for i in range(n_examples):
        # compute similarity
        sim_raw[i], sim_lca[i] = compute_cell_memory_similarity_i(
            C[i], V[i][0], em_gate[i], cmpt)
    return sim_raw, sim_lca


def compute_cell_memory_similarity_i(
        C_i, V_i, em_gate_i, cmpt, leak=0, kernel='cosine', recall_func='LCA'
):
    n_timepoints = np.shape(C_i)[0]
    n_memories = len(V_i)
    sim_raw_i = np.zeros((n_timepoints, n_memories))
    sim_lca_i = np.zeros((n_timepoints, n_memories))
    for t in range(n_timepoints):
        # compute raw similarity
        sim_raw_i[t, :] = to_np(compute_similarities(
            to_pth(C_i[t]), V_i, kernel))
        # compute LCA similarity
        sim_lca_i[t, :] = transform_similarities(
            to_pth(sim_raw_i[t, :]), recall_func,
            leak=leak, comp=cmpt, w_input=em_gate_i[t]
        )
    return sim_raw_i, sim_lca_i


def nansmooth_mean(mat, axis, N=10):
    return np.convolve(np.nanmean(mat, axis=axis), np.ones(N) / N, mode='valid')


def sep_data(data, cond):
    data_cong = deepcopy(data)
    data_incong = deepcopy(data)
    data_cong = data_cong[cond == 1]
    data_incong = data_incong[cond == 0]
    return data_cong, data_incong


# def make_bars(log_data, log_condition, n_cue, f, ax):
#     import seaborn as sns
#     cpals = sns.color_palette()
#     d_cong_tst, d_incong_tst = sep_data(
#         log_data[:, n_cue:], log_condition[:, 1, :])
#     d_cong_std, d_incong_std = sep_data(
#         log_data[:, :n_cue], log_condition[:, 0, :])

#     d_list = [
#         d_cong_tst, d_incong_tst,
#         # d_cong_std, d_incong_std
#     ]
#     lgds = [
#         'congruent', 'incongruent',
#         # 'schematic, test', 'control, test',
#         # 'schematic, study', 'control, study'
#     ]
#     height = [np.nanmean(d_i) for d_i in d_list]
#     # print(np.shape(sem(d_cong_tst, nan_policy='omit', axis=0)))
#     errorbars = np.array([
#         np.nanmean(sem(d_i, nan_policy='omit', axis=0))
#         for d_i in d_list
#     ])

#     ax.bar(
#         x=range(len(lgds)), yerr=errorbars, height=height,
#         color=cpals[:len(d_list)]
#     )
#     ax.set_ylim([0, 1.05])
#     ax.set_xticks(range(len(lgds)))
#     ax.set_xticklabels(lgds)
#     sns.despine()
#     return f, ax
