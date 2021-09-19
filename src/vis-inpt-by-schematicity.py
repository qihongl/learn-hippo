# import dabest
# import pandas as pd
import os
import numpy as np
import numpy.ma as ma
from utils.params import P
from utils.io import pickle_load_dict, build_log_path
from utils.constants import TZ_COND_DICT
from analysis import compute_stats, remove_none
from scipy.stats import pearsonr
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='poster')
log_root = '../log/'
# log_root = '/Volumes/Extreme SSD 1/research/'
# constants
lca_pnames = {0: 'input gate', 1: 'competition'}
all_conds = list(TZ_COND_DICT.values())
T = 16

exp_name = 'vary-schema-level-prandom'
# exp_name = 'vary-schema-level-prandom-ndk'
# exp_name = 'vary-schema-level-after-ig.3'
clip_subj = 15
comp_val = .8

n_subjs = 15
subj_ids = np.arange(n_subjs)
n_def_tps = 8
# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0
attach_cond = 0
supervised_epoch = 600
epoch_load = 1000
learning_rate = 7e-4
n_branch = 4
n_param = 16
enc_size = 16
n_event_remember = 2
similarity_max_test = .9
similarity_min_test = 0
n_examples_test = 256

penalty_random = 1
# penalty_train_test_label = [
#     'training penalty = 0', 'training penalty = 2', 'training penalty = 4'
# ]
# penalty_train_test = [[0, 0], [2, 2], [4, 4]]

penalty_train_test_label = [
    'test penalty = 0', 'test penalty = 2', 'test penalty = 4'
]
penalty_train_test = [[4, 0], [4, 2], [4, 4]]
# penalty_train_test = [[4, 2]]

def_tps_g_byp = [[] for _ in range(len(penalty_train_test))]
def_prob_range = np.arange(.25, 1, .1)
n_schema_lvs = len(def_prob_range)
n_penalty_lvs = len(penalty_train_test)

input_s_mu = np.zeros((n_penalty_lvs, n_schema_lvs))
input_ns_mu = np.zeros((n_penalty_lvs, n_schema_lvs))
input_s_se = np.zeros((n_penalty_lvs, n_schema_lvs))
input_ns_se = np.zeros((n_penalty_lvs, n_schema_lvs))
tma_s_mu = np.zeros((n_penalty_lvs, n_schema_lvs))
tma_ns_mu = np.zeros((n_penalty_lvs, n_schema_lvs))
tma_s_se = np.zeros((n_penalty_lvs, n_schema_lvs))
tma_ns_se = np.zeros((n_penalty_lvs, n_schema_lvs))

for pi, (penalty_train, penalty_test) in enumerate(penalty_train_test):

    '''load data'''

    for dpi, def_prob in enumerate(def_prob_range):

        p = P(
            exp_name=exp_name, sup_epoch=supervised_epoch,
            n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
            enc_size=enc_size, n_event_remember=n_event_remember,
            def_prob=def_prob, n_def_tps=n_def_tps,
            penalty=penalty_train, penalty_random=penalty_random,
            p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
            cmpt=comp_val,
        )

        fname = 'p%d-%d.pkl' % (penalty_train, penalty_test)
        # pickle_save_dict(gdata_dict, os.path.join(dir_all_subjs, fname))
        log_path, _ = build_log_path(
            0, p, log_root=log_root, mkdir=False, verbose=False
        )
        data_load_path = os.path.join(os.path.dirname(log_path), fname)
        # load data
        # fname = '%s-dp%.2f-p%d-%d.pkl' % (
        #     exp_name, def_prob, penalty_train, penalty_test)
        # print(fname)
        # data_load_path = os.path.join(gdata_outdir, fname)
        data = pickle_load_dict(data_load_path)
        # unpack data
        inpt_dmp2_g = remove_none(data['inpt_dmp2_g'])
        actions_dmp2_g = remove_none(data['actions_dmp2_g'])
        targets_dmp2_g = remove_none(data['targets_dmp2_g'])
        def_path_int_g = remove_none(data['def_path_int_g'])
        def_tps_g = remove_none(data['def_tps_g'])
        ma_g = remove_none(data['lca_ma_list'])

        # clip the number of subjects
        n_rm = len(def_tps_g) - clip_subj
        if n_rm > 0:
            inpt_dmp2_g = inpt_dmp2_g[: len(def_tps_g) - n_rm]
            actions_dmp2_g = actions_dmp2_g[: len(def_tps_g) - n_rm]
            targets_dmp2_g = targets_dmp2_g[: len(def_tps_g) - n_rm]
            def_path_int_g = def_path_int_g[: len(def_tps_g) - n_rm]
            def_tps_g = def_tps_g[: len(def_tps_g) - n_rm]
            ma_g = ma_g[: len(def_tps_g) - n_rm]

        def_tps_g = [np.array(def_tps).astype(np.bool)
                     for def_tps in def_tps_g]
        def_tps_g_byp[pi].append(def_tps_g)
        actual_n_subjs = len(def_path_int_g)

        # get the target memory activation for each subject
        tma_dmp2_g = np.array([ma_g[i_s]['DM']['targ']['mu'][n_param:]
                              for i_s in range(actual_n_subjs)])

        # split the data according to whether t is schematic ...

        # ... for target memory activation
        tma_s = ma.masked_array(tma_dmp2_g, np.logical_not(def_tps_g))
        tma_ns = ma.masked_array(tma_dmp2_g, def_tps_g)
        # compute mean
        tma_s_mu[pi, dpi], tma_s_se[pi, dpi] = compute_stats(
            np.mean(tma_s, axis=1))
        tma_ns_mu[pi, dpi], tma_ns_se[pi, dpi] = compute_stats(
            np.mean(tma_ns, axis=1))

        # ... for the em gates
        inpt_s_l, inpt_ns_l = [[] for t in range(T)], [[] for t in range(T)]
        ms_s_l, ms_ns_l = [[] for t in range(T)], [[] for t in range(T)]

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

        inpt_s_t_mu, inpt_s_t_se = np.zeros(T,), np.zeros(T,)
        inpt_ns_t_mu, inpt_ns_t_se = np.zeros(T,), np.zeros(T,)

        for t in range(T):
            inpt_s_t_mu[t], inpt_s_t_se[t] = compute_stats(inpt_s_l[t])
            inpt_ns_t_mu[t], inpt_ns_t_se[t] = compute_stats(inpt_ns_l[t])

        input_s_mu[pi, dpi], input_s_se[pi, dpi] = compute_stats(
            np.concatenate(inpt_s_l))
        input_ns_mu[pi, dpi], input_ns_se[pi, dpi] = compute_stats(
            np.concatenate(inpt_ns_l))

'''make plots - EM gate and target memory activation across schema levels'''

f1, axes1 = plt.subplots(1, 3, figsize=(21, 5))
f2, axes2 = plt.subplots(1, 3, figsize=(21, 5))

for pi, (penalty_train, penalty_test) in enumerate(penalty_train_test):
    # EM gate across schema levels
    axes1[pi].errorbar(x=range(n_schema_lvs), y=input_ns_mu[pi],
                       yerr=input_ns_se[pi], label='non schematic')
    axes1[pi].errorbar(x=range(n_schema_lvs), y=input_s_mu[pi],
                       yerr=input_s_se[pi], label='schematic')
    # axes1[pi].set_title(penalty_train_test_label[pi])
    axes1[pi].set_xticks(range(n_schema_lvs))
    axes1[pi].set_xticklabels(['%.2f' % dp for dp in def_prob_range])
    axes1[pi].set_xlabel('Schema strength')
    axes1[pi].set_ylabel('Average EM gate')
    # target memory activation across schema levels
    axes2[pi].errorbar(x=range(n_schema_lvs), y=tma_ns_mu[pi],
                       yerr=tma_ns_se[pi], label='non schematic')
    axes2[pi].errorbar(x=range(n_schema_lvs), y=tma_s_mu[pi],
                       yerr=tma_s_se[pi], label='schematic')
    # axes2[pi].set_title(penalty_train_test_label[pi])
    axes2[pi].set_xticks(range(n_schema_lvs))
    axes2[pi].set_xticklabels(['%.2f' % dp for dp in def_prob_range])
    axes2[pi].set_xlabel('Schema strength')
    axes2[pi].set_ylabel('Memory activation')
# mark the plots

axes1[1].legend()
f1.tight_layout()
sns.despine(fig=f1)

axes2[1].legend()
f2.tight_layout()
sns.despine(fig=f2)

img_name = 'ig-characteristics-by-schematicity.png'
f1.savefig(os.path.join('../figs', img_name), bbox_inches='tight')

img_name = 'tma-characteristics-by-schematicity.png'
f2.savefig(os.path.join('../figs', img_name), bbox_inches='tight')

# '''make plots - EM gate just for the intermediate penalty level'''


f, ax = plt.subplots(1, 1, figsize=(7, 5))
penalty_train, penalty_test, pi = 4, 2, 1

# EM gate across schema levels
ax.errorbar(x=range(n_schema_lvs), y=input_ns_mu[pi],
            yerr=input_ns_se[pi], label='non schematic')
ax.errorbar(x=range(n_schema_lvs), y=input_s_mu[pi],
            yerr=input_s_se[pi], label='schematic')
# axes1[pi].set_title(penalty_train_test_label[pi])
ax.set_xticks(range(n_schema_lvs))
ax.set_xticklabels(['%.2f' % dp for dp in def_prob_range])
ax.set_xlabel('Schema strength')
ax.set_ylabel('Average EM gate')
ax.legend()
f.tight_layout()
sns.despine()
