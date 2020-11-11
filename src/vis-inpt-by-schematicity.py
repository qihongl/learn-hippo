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
exp_name = '1029-schema-evenodd'
gdata_outdir = 'temp/'
# n_subjs = 15

# penalty_train = 0
# penalty_test = 0
f, axes = plt.subplots(1, 3, figsize=(23, 5), sharey=True)
penalty_train_test_label = ['training penalty = 0',
                            'training penalty = 4', 'training penalty = 8']
penalty_train_test = [[0, 0], [4, 2], [8, 4]]

for pi, (penalty_train, penalty_test) in enumerate(penalty_train_test):

    def_prob_range = np.arange(.25, 1, .1)
    n_def_prob = len(def_prob_range)

    '''load data'''
    inpt_mu = [np.zeros(T,) for dpi in range(n_def_prob)]
    inpt_selectivity = [[] for dpi in range(n_def_prob)]
    rt_mu, rt_se = np.zeros(n_def_prob,), np.zeros(n_def_prob,)
    input_s_mu = [np.zeros(T,) for dpi in range(n_def_prob)]
    input_ns_mu = [np.zeros(T,) for dpi in range(n_def_prob)]
    # input_sc_mu = [np.zeros(T,) for dpi in range(n_def_prob)]
    # input_sv_mu = [np.zeros(T,) for dpi in range(n_def_prob)]

    input_s_se = [np.zeros(T,) for dpi in range(n_def_prob)]
    input_ns_se = [np.zeros(T,) for dpi in range(n_def_prob)]
    # input_sc_se = [np.zeros(T,) for dpi in range(n_def_prob)]
    # input_sv_se = [np.zeros(T,) for dpi in range(n_def_prob)]
    input_s_totalmu = np.zeros(n_def_prob,)
    input_ns_totalmu = np.zeros(n_def_prob,)
    input_s_totalse = np.zeros(n_def_prob,)
    input_ns_totalse = np.zeros(n_def_prob,)
    # for ptrain in penaltys_train:
    # ptrain = penaltys_train[0]

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
        def_tps_g = [np.array(def_tps).astype(np.bool)
                     for def_tps in def_tps_g]
        actual_n_subjs = len(def_path_int_g)
        print(actual_n_subjs)

        input_ns, input_sc, input_sv = [], [], []
        mask_ns, mask_sc, mask_sv = [], [], []
        # actual_n_subjs
        inpt_s_l = [[] for t in range(T)]
        inpt_ns_l = [[] for t in range(T)]
        inpt_s = np.zeros(T,)
        inpt_ns = np.zeros(T,)
        mask_s_cum = np.zeros(T,)
        mask_ns_cum = np.zeros(T,)
        for i_s in range(actual_n_subjs):
            n_trials = np.shape(targets_dmp2_g[i_s])[0]
            # compute the average input gate for the i-th subject
            inpt_mu_i_s, inpt_se_i_s = compute_stats(inpt_dmp2_g[i_s])
            inpt_mu[dpi] += inpt_mu_i_s
            # compute rt
            rt_i_s = np.dot(inpt_dmp2_g[i_s], np.arange(T))
            # compute the proto time point mask for subj i
            def_path_int_i_s_rep = np.tile(def_path_int_g[i_s], (n_trials, 1))
            def_tps_g_i_s_rep = np.tile(def_tps_g[i_s], (n_trials, 1))
            sc_tp_i_s = np.logical_and(
                (targets_dmp2_g[i_s] == def_path_int_i_s_rep), def_tps_g_i_s_rep)
            sv_tp_i_s = np.logical_and(
                (targets_dmp2_g[i_s] != def_path_int_i_s_rep), def_tps_g_i_s_rep)

            # compute input gate selecitivity: non-schematic tp - schematic tp
            mask_s = def_tps_g_i_s_rep
            inpt_s_ = inpt_dmp2_g[i_s][:, def_tps_g[i_s]]
            inpt_ns_ = inpt_dmp2_g[i_s][:, ~def_tps_g[i_s]]
            # inpt_selectivity_i_s = np.mean(inpt_ns_) - np.mean(inpt_s_)
            # inpt_selectivity[dpi].append(inpt_selectivity_i_s)
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

        input_s_mu[dpi] = inpt_s_mu
        input_ns_mu[dpi] = inpt_ns_mu
        input_s_se[dpi] = inpt_s_mu
        input_ns_se[dpi] = inpt_ns_mu

        # np.shape(np.concatenate(inpt_s_l))
        input_s_totalmu[dpi], input_s_totalse[dpi] = compute_stats(
            np.concatenate(inpt_s_l))
        input_ns_totalmu[dpi], input_ns_totalse[dpi] = compute_stats(
            np.concatenate(inpt_ns_l))

        # plt.errorbar(x=range(T), y=inpt_s_mu, yerr=inpt_s_se)
        # plt.errorbar(x=range(T), y=inpt_ns_mu, yerr=inpt_ns_se)

        #     # compute mean input gate for 1) non schematic
        #     # for 2.1) schema consistent, 2.2) schema violated
        #     input_ns_i_s = np.zeros((n_trials, T))
        #     input_sc_i_s = np.zeros((n_trials, T))
        #     input_sv_i_s = np.zeros((n_trials, T))
        #     mask_ns.append(~def_tps_g_i_s_rep)
        #     mask_sc.append(sc_tp_i_s)
        #     mask_sv.append(sv_tp_i_s)
        #     input_ns_i_s[~def_tps_g_i_s_rep] += inpt_dmp2_g[i_s][~def_tps_g_i_s_rep]
        #     input_sc_i_s[sc_tp_i_s] += inpt_dmp2_g[i_s][sc_tp_i_s]
        #     input_sv_i_s[sv_tp_i_s] += inpt_dmp2_g[i_s][sv_tp_i_s]
        #     input_ns.append(np.sum(input_ns_i_s, axis=0))
        #     input_sc.append(np.sum(input_sc_i_s, axis=0))
        #     input_sv.append(np.sum(input_sv_i_s, axis=0))
        #
        # input_ns_sum = np.sum(input_ns, axis=0)
        # input_sc_sum = np.sum(input_sc, axis=0)
        # input_sv_sum = np.sum(input_sv, axis=0)
        # input_ns_std = np.std(input_ns, axis=0)
        # input_sc_std = np.std(input_sc, axis=0)
        # input_sv_std = np.std(input_sv, axis=0)
        #
        # input_ns_mu[dpi] = input_ns_sum / np.sum(np.vstack(mask_ns), axis=0)
        # input_sc_mu[dpi] = input_sc_sum / np.sum(np.vstack(mask_sc), axis=0)
        # input_sv_mu[dpi] = input_sv_sum / np.sum(np.vstack(mask_sv), axis=0)
        # input_s_mu[dpi] = (input_sc_mu[dpi] + input_sv_mu[dpi]) / 2
        #
        # input_ns_se[dpi] = input_ns_std / \
        #     np.sqrt(np.sum(np.vstack(mask_ns), axis=0)) / \
        #     np.sum(np.vstack(mask_ns), axis=0)
        # input_sc_se[dpi] = input_sc_std / \
        #     np.sqrt(np.sum(np.vstack(mask_sc), axis=0)) / \
        #     np.sqrt(np.sum(np.vstack(mask_sc), axis=0))
        # input_sv_se[dpi] = input_sv_std / \
        #     np.sqrt(np.sum(np.vstack(mask_sv), axis=0)) / \
        #     np.sqrt(np.sum(np.vstack(mask_sv), axis=0))
        #
        # # compute average input gate
        # inpt_mu[dpi] /= actual_n_subjs
        # rt_mu[dpi], rt_se[dpi] = compute_stats(rt_i_s)

    '''plot'''

    # average input gate ~ schema level
    # bpal = sns.color_palette("Blues", n_colors=n_def_prob)
    # f, ax = plt.subplots(1, 1, figsize=(8, 6))
    # for dpi, def_prob in enumerate(def_prob_range):
    #     ax.plot(inpt_mu[dpi], color=bpal[dpi], label='%.2f' % def_prob)
    # ax.set_ylabel('Input gate')
    # ax.set_xlabel('Time')
    # ax.legend()
    # sns.despine()

    # # input gate for the 3 condition for each schema level
    # for dpi, def_prob in enumerate(def_prob_range):
    #     f, ax = plt.subplots(1, 1, figsize=(5, 4))
    #     ax.plot(input_ns_mu[dpi], label='ns')
    #     ax.plot(input_sc_mu[dpi], label='sc')
    #     ax.plot(input_sv_mu[dpi], label='sv')
    #     ax.set_title('Schema strenght = %.2f ' % def_prob)
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Input gate')
    #     ax.set_ylim([-.05, .7])
    #     ax.legend()
    #     sns.despine()

    #
    # # input gate for the 3 condition for each schema level
    # for dpi, def_prob in enumerate(def_prob_range):
    #     f, ax = plt.subplots(1, 1, figsize=(5, 4))
    #     ax.plot(input_ns_mu[dpi], label='ns')
    #     ax.plot(input_s_mu[dpi], label='s')
    #     ax.set_title('Schema strenght = %.2f ' % def_prob)
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Input gate')
    #     ax.set_ylim([-.05, .7])
    #     ax.legend()
    #     sns.despine()
    #
    # summarize by max
    # input gate for the 3 condition for each schema level
    # max_inpt_ns = [np.mean(input_ns_mu[dpi]) for dpi in range(n_def_prob)]
    # max_inpt_s = [np.mean(input_s_mu[dpi]) for dpi in range(n_def_prob)]

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

# ax.set_ylim([-.05, .7])
#
#
# effect = np.zeros((n_def_prob, T))
# effect_se = np.zeros((n_def_prob, T))
# bpal = sns.color_palette("Blues", n_colors=n_def_prob)
# f, ax = plt.subplots(1, 1, figsize=(8, 5))
# ax.axhline(0, linestyle='--', color='k', alpha=.3)
# for dpi, def_prob in enumerate(def_prob_range):
#     # effect[dpi, :] = (input_ns_mu[dpi] + input_sv_mu[dpi]) / \
#     #     2 - input_sc_mu[dpi]
#     effect[dpi, :] = input_ns_mu[dpi] - \
#         (input_sc_mu[dpi] + input_sv_mu[dpi]) / 2
#     effect_se[dpi, :] = input_sv_se[dpi] + input_sc_se[dpi]
#     ax.errorbar(
#         x=range(T), y=effect[dpi, :],
#         yerr=effect_se[dpi, :],
#         color=bpal[dpi], label='%.2f' % def_prob
#     )
# ax.set_title('Schema consistent recall supression')
# ax.set_ylabel('Input gate difference')
# ax.set_xlabel('Time')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# sns.despine()
#
# f, ax = plt.subplots(1, 1, figsize=(7, 5))
# ax.errorbar(x=range(n_def_prob), y=np.sum(
#     effect, axis=1), yerr=np.sum(effect_se, axis=1) * 1.96)
# ax.axhline(0, linestyle='--', color='k', alpha=.3)
# ax.set_xticks(range(n_def_prob))
# ax.set_xticklabels(['%.2f' % dp for dp in def_prob_range])
# ax.set_xlabel('Schema strength')
# ax.set_title('Area under curve')
# sns.despine()
#
# # recall time ~ schema level
# f, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.bar(x=range(n_def_prob), height=rt_mu, yerr=rt_se)
# ax.set_xticks(range(n_def_prob))
# ax.set_xticklabels(['%.2f' % dp for dp in def_prob_range])
# ax.set_xlabel('Schema strength')
# ax.set_ylabel('RT')
# sns.despine()
#
#
# # compute the difference between inpt at schematic vs. nonschematic tps
# # call the difference schematic selectivity
# inpt_selectivity_mu = np.array([np.mean(temp) for temp in inpt_selectivity])
# inpt_selectivity_se = np.array(
#     [np.std(temp) / np.sqrt(len(temp)) for temp in inpt_selectivity])
# f, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.axhline(0, linestyle='--', color='k', alpha=.3)
# ax.errorbar(x=range(n_def_prob), y=inpt_selectivity_mu,
#             yerr=inpt_selectivity_se)
# ax.set_xticks(range(n_def_prob))
# ax.set_xticklabels(['%.2f' % dp for dp in def_prob_range])
# ax.set_xlabel('Schema strength')
# ax.set_ylabel('Schematic selectivity')
# sns.despine()
