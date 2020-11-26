# import dabest
# import pandas as pd
import os
import numpy as np
from utils.io import pickle_load_dict
from utils.constants import TZ_COND_DICT
from analysis import compute_stats, remove_none
from scipy.stats import pearsonr
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='poster')

# constants
lca_pnames = {0: 'input gate', 1: 'competition'}
all_conds = list(TZ_COND_DICT.values())
T = 16

# exp_name = '0916-widesim-prandom-schema'
exp_name = '1029-schema-evenodd-pfixed'
# exp_name = '1029-schema-evenodd'
gdata_outdir = 'data/'
clip_subj = 15

f, axes = plt.subplots(1, 3, figsize=(23, 5), sharey=True)
penalty_train_test_label = [
    'training penalty = 0', 'training penalty = 2', 'training penalty = 4'
]
penalty_train_test = [[0, 0], [2, 2], [4, 4]]
def_tps_g_byp = [[] for _ in range(len(penalty_train_test))]
def_prob_range = np.arange(.25, 1, .1)
n_def_prob = len(def_prob_range)

for pi, (penalty_train, penalty_test) in enumerate(penalty_train_test):

    '''load data'''
    input_s_totalmu = np.zeros(n_def_prob,)
    input_ns_totalmu = np.zeros(n_def_prob,)
    input_s_totalse = np.zeros(n_def_prob,)
    input_ns_totalse = np.zeros(n_def_prob,)

    for dpi, def_prob in enumerate(def_prob_range):
        # load data
        fname = '%s-dp%.2f-p%d-%d.pkl' % (
            exp_name, def_prob, penalty_train, penalty_test)
        print(fname)
        data_load_path = os.path.join(gdata_outdir, fname)
        data = pickle_load_dict(data_load_path)
        # unpack data
        inpt_dmp2_g = remove_none(data['inpt_dmp2_g'])
        actions_dmp2_g = remove_none(data['actions_dmp2_g'])
        targets_dmp2_g = remove_none(data['targets_dmp2_g'])
        def_path_int_g = remove_none(data['def_path_int_g'])
        def_tps_g = remove_none(data['def_tps_g'])

        n_rm = len(def_tps_g) - clip_subj
        if n_rm > 0:
            inpt_dmp2_g = inpt_dmp2_g[: len(def_tps_g) - n_rm]
            actions_dmp2_g = actions_dmp2_g[: len(def_tps_g) - n_rm]
            targets_dmp2_g = targets_dmp2_g[: len(def_tps_g) - n_rm]
            def_path_int_g = def_path_int_g[: len(def_tps_g) - n_rm]
            def_tps_g = def_tps_g[: len(def_tps_g) - n_rm]

        def_tps_g = [np.array(def_tps).astype(np.bool)
                     for def_tps in def_tps_g]
        def_tps_g_byp[pi].append(def_tps_g)
        actual_n_subjs = len(def_path_int_g)

        inpt_s_l = [[] for t in range(T)]
        inpt_ns_l = [[] for t in range(T)]
        for i_s in range(actual_n_subjs):
            n_trials = np.shape(targets_dmp2_g[i_s])[0]
            # compute the proto time point mask for subj i
            def_path_int_i_s_rep = np.tile(def_path_int_g[i_s], (n_trials, 1))
            def_tps_g_i_s_rep = np.tile(def_tps_g[i_s], (n_trials, 1))
            mask_s = def_tps_g_i_s_rep
            inpt_s_ = inpt_dmp2_g[i_s][:, def_tps_g[i_s]]
            inpt_ns_ = inpt_dmp2_g[i_s][:, ~def_tps_g[i_s]]
            for t in range(np.shape(inpt_s_)[1]):
                abs_tid = np.where(mask_s[0] == True)[0][t]
                inpt_s_l[abs_tid].extend(list(inpt_s_[:, t]))
            for t in range(np.shape(inpt_ns_)[1]):
                abs_tid = np.where(~mask_s[0] == True)[0][t]
                inpt_ns_l[abs_tid].extend(list(inpt_ns_[:, t]))

        inpt_s_mu, inpt_s_se = np.zeros(T,), np.zeros(T,)
        inpt_ns_mu, inpt_ns_se = np.zeros(T,), np.zeros(T,)

        for t in range(T):
            inpt_s_mu[t], inpt_s_se[t] = compute_stats(inpt_s_l[t])
            inpt_ns_mu[t], inpt_ns_se[t] = compute_stats(inpt_ns_l[t])

        input_s_totalmu[dpi], input_s_totalse[dpi] = compute_stats(
            np.concatenate(inpt_s_l))
        input_ns_totalmu[dpi], input_ns_totalse[dpi] = compute_stats(
            np.concatenate(inpt_ns_l))

    axes[pi].errorbar(x=range(n_def_prob), y=input_ns_totalmu,
                      yerr=input_ns_totalse, label='non schematic')
    axes[pi].errorbar(x=range(n_def_prob), y=input_s_totalmu,
                      yerr=input_s_totalse, label='schematic')
    axes[pi].set_title(penalty_train_test_label[pi])
    axes[pi].set_xticks(range(n_def_prob))
    axes[pi].set_xticklabels(['%.2f' % dp for dp in def_prob_range])
    axes[pi].set_xlabel('Schema level')
axes[0].set_ylabel('Average input gate')
axes[0].legend()
sns.despine()
img_name = 'ig-characteristics-by-schematicity.png'
f.savefig(os.path.join('../figs', img_name), bbox_inches='tight')

# for pi, (penalty_train, penalty_test) in enumerate(penalty_train_test):
#     print(pi, len(def_tps_g_byp[pi]))
#     for dpi, def_prob in enumerate(def_prob_range):
#         print(def_prob_range[dpi], len(def_tps_g_byp[pi][dpi]))
#         np.shape(np.sum(np.array(def_tps_g_byp[pi][dpi]), axis=0))
#         print(np.sum(np.array(def_tps_g_byp[pi][dpi]), axis=0))
