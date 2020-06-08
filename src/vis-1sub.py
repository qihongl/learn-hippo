import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings

from itertools import product
from scipy.stats import pearsonr
from sklearn import metrics
from task import SequenceLearning
from copy import deepcopy
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.utils import find_factors
from utils.io import build_log_path, get_test_data_dir, \
    pickle_load_dict, get_test_data_fname, pickle_save_dict, load_env_metadata
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_cell_memory_similarity, create_sim_dict, compute_mistake, \
    compute_event_similarity, batch_compute_true_dk, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip,\
    compute_cell_memory_similarity_stats, sep_by_qsource, prop_true, \
    get_qsource, trim_data, compute_roc, get_hist_info, remove_none
from analysis.task import get_oq_keys
from vis import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA

warnings.filterwarnings("ignore")
# plt.switch_backend('agg')
sns.set(style='white', palette='colorblind', context='poster')

all_conds = TZ_COND_DICT.values()

log_root = '../log/'
# exp_name = '0429-widesim-attachcond'
# exp_name = '0220-v1-widesim-comp.8'
exp_name = '0425-schema.7-comp.8'

supervised_epoch = 600
epoch_load = 1000
learning_rate = 7e-4

n_branch = 4
n_param = 16
enc_size = 16
n_event_remember = 2

# def_prob = None
# n_def_tps = 0
def_prob = .7
n_def_tps = 8

comp_val = .8
leak_val = 0

n_hidden = 194
n_hidden_dec = 128
# n_hidden = 256
# n_hidden_dec = 194
eta = .1

# testing param, ortho to the training directory
penalty_random = 1
penalty_discrete = 1
penalty_onehot = 0
normalize_return = 1

# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0

# testing params
enc_size_test = 16

pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
slience_recall_time = None
# slience_recall_time = range(n_param)

# similarity_max_test = .9
# similarity_min_test = .35
similarity_max_test = .9
similarity_min_test = 0
n_examples_test = 256

# subj_ids = [9]
subj_ids = np.arange(9)

# penaltys_train = [0, 4]
# penaltys_test = np.array([0, 2, 4])
penaltys_test = np.array([2])
penaltys_train = [4]
# penaltys_test = np.array([0, 2, 4])

# penaltys_train = [0, 1, 2, 4, 8]
# penaltys_test = np.array([0, 1, 2, 4, 8])

n_subjs = len(subj_ids)
DM_qsources = ['EM only', 'both']

if not os.path.isdir(f'../figs/{exp_name}'):
    os.makedirs(f'../figs/{exp_name}')


def prealloc_stats():
    return {cond: {'mu': [None] * n_subjs, 'er': [None] * n_subjs}
            for cond in all_conds}


for penalty_train in penaltys_train:
    penaltys_test_ = penaltys_test[penaltys_test <= penalty_train]
    # print(penalty_train, penaltys_test_)
    for penalty_test in penaltys_test_:
        # penalty_train, penalty_test = 0, 0
        print(f'penalty_train={penalty_train}, penalty_test={penalty_test}')

        acc_dict = prealloc_stats()
        mis_dict = prealloc_stats()
        dk_dict = prealloc_stats()
        inpt_dict = prealloc_stats()
        leak_dict = prealloc_stats()
        comp_dict = prealloc_stats()
        inpt_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        leak_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        comp_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        ma_raw_list = [None] * n_subjs
        ma_list = [None] * n_subjs
        ma_cos_list = [None] * n_subjs
        tma_dm_p2_dict_bq = {qs: [None] * n_subjs for qs in DM_qsources}
        q_source_list = [None] * n_subjs
        ms_lure_list = [None] * n_subjs
        ms_targ_list = [None] * n_subjs
        tpr_list = [None] * n_subjs
        fpr_list = [None] * n_subjs
        auc_list = [None] * n_subjs
        # cmpt_bar_list = [None] * n_subjs
        def_tps_list = [None] * n_subjs

        inpt_wproto_c_g = [None] * n_subjs
        inpt_wproto_ic_g = [None] * n_subjs
        inpt_woproto_c_g = [None] * n_subjs
        inpt_woproto_ic_g = [None] * n_subjs
        corrects_dmp2_wproto_c_g = [None] * n_subjs
        corrects_dmp2_wproto_ic_g = [None] * n_subjs
        corrects_dmp2_woproto_c_g = [None] * n_subjs
        corrects_dmp2_woproto_ic_g = [None] * n_subjs
        corrects_dmp2_wwoproto_cic_g = np.zeros((n_subjs, 2, 2))
        inpt_wwoproto_cic_g = np.zeros((n_subjs, 2, 2))
        dk_wwoproto_cic_g = np.zeros((n_subjs, 2, 2))
        n_sc_mistakes_g = np.zeros(n_subjs)
        n_sic_mistakes_g = np.zeros(n_subjs)
        n_corrects_g = np.zeros(n_subjs)
        n_dks_g = np.zeros(n_subjs)

        for i_s, subj_id in enumerate(subj_ids):
            np.random.seed(subj_id)
            torch.manual_seed(subj_id)

            '''init'''
            p = P(
                exp_name=exp_name, sup_epoch=supervised_epoch,
                n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
                enc_size=enc_size, n_event_remember=n_event_remember,
                def_prob=def_prob, n_def_tps=n_def_tps,
                penalty=penalty_train, penalty_random=penalty_random,
                penalty_onehot=penalty_onehot, penalty_discrete=penalty_discrete,
                normalize_return=normalize_return,
                p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
                n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
                lr=learning_rate, eta=eta,
            )
            # create logging dirs
            test_params = [penalty_test, pad_len_test, slience_recall_time]
            log_path, log_subpath = build_log_path(
                subj_id, p, log_root=log_root, mkdir=False)
            env = load_env_metadata(log_subpath)
            def_path = np.array(env['def_path'])

            def_tps = env['def_tps']
            def_tps_list[i_s] = def_tps
            log_subpath['data']
            print(log_subpath['data'])
            p.update_enc_size(enc_size_test)

            # init env
            task = SequenceLearning(
                n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
                p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
                def_path=def_path, def_prob=def_prob, def_tps=def_tps,
                similarity_cap_lag=p.n_event_remember,
                similarity_max=similarity_max_test,
                similarity_min=similarity_min_test
            )

            test_data_dir, test_data_subdir = get_test_data_dir(
                log_subpath, epoch_load, test_params)
            test_data_fname = get_test_data_fname(n_examples_test)
            fpath = os.path.join(test_data_dir, test_data_fname)
            if not os.path.exists(fpath):
                print('DNE')
                # print(fpath)
                continue

            test_data_dict = pickle_load_dict(fpath)
            results = test_data_dict['results']
            XY = test_data_dict['XY']

            [dist_a_, Y_, log_cache_, log_cond_] = results
            [X_raw, Y_raw] = XY
            # np.shape(X_raw)
            # np.shape(Y_)

            # compute ground truth / objective uncertainty (delay phase removed)
            true_dk_wm_, true_dk_em_ = batch_compute_true_dk(X_raw, task)

            '''precompute some constants'''
            # figure out max n-time-steps across for all trials
            T_part = n_param + pad_len_test
            T_total = T_part * task.n_parts
            #
            n_conds = len(TZ_COND_DICT)
            memory_types = ['targ', 'lure']
            ts_predict = np.array(
                [t % T_part >= pad_len_test for t in range(T_total)])

            '''organize results to analyzable form'''
            # skip examples untill EM is full
            n_examples_skip = compute_n_trials_to_skip(log_cond_, p)
            n_examples = n_examples_test - n_examples_skip
            data_to_trim = [dist_a_, Y_, log_cond_,
                            log_cache_, true_dk_wm_, true_dk_em_, X_raw]
            [dist_a, Y, log_cond, log_cache, true_dk_wm, true_dk_em, X_raw] = trim_data(
                n_examples_skip, data_to_trim)
            # process the data
            n_trials = np.shape(X_raw)[0]
            trial_id = np.arange(n_trials)
            cond_ids = get_trial_cond_ids(log_cond)

            activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
            [C, H, M, CM, DA, V] = activity_
            [inpt] = ctrl_param_

            comp = np.full(np.shape(inpt), comp_val)
            leak = np.full(np.shape(inpt), leak_val)

            # onehot to int
            actions = np.argmax(dist_a, axis=-1)
            targets = np.argmax(Y, axis=-1)

            # compute performance
            corrects = targets == actions
            dks = actions == p.dk_id
            mistakes = np.logical_and(targets != actions, ~dks)

            # split data wrt p1 and p2
            CM_p1, CM_p2 = CM[:, :T_part, :], CM[:, T_part:, :]
            DA_p1, DA_p2 = DA[:, :T_part, :], DA[:, T_part:, :]
            X_raw_p1 = np.array(X_raw)[:, :T_part, :]
            X_raw_p2 = np.array(X_raw)[:, T_part:, :]
            #
            corrects_p2 = corrects[:, T_part:]
            mistakes_p1 = mistakes[:, :T_part]
            mistakes_p2 = mistakes[:, T_part:]
            dks_p2 = dks[:, T_part:]
            inpt_p2 = inpt[:, T_part:]
            # np.shape(inpt_p2)
            # plt.plot(inpt_p2[:10,:].T)
            targets_p1, targets_p2 = targets[:, :T_part], targets[:, T_part:]
            actions_p1, actions_p2 = actions[:, :T_part], actions[:, T_part:]

            # pre-extract p2 data for the DM condition
            corrects_dmp2 = corrects_p2[cond_ids['DM']]
            mistakes_dmp2 = mistakes_p2[cond_ids['DM']]
            mistakes_dmp1 = mistakes_p1[cond_ids['DM']]
            dks_dmp2 = dks_p2[cond_ids['DM']]
            CM_dmp2 = CM_p2[cond_ids['DM']]
            DA_dmp2 = DA_p2[cond_ids['DM']]

            inpt_dmp2 = inpt_p2[cond_ids['DM']]
            targets_dmp2 = targets_p2[cond_ids['DM'], :]
            actions_dmp2 = actions_p2[cond_ids['DM']]
            targets_dmp1 = targets_p1[cond_ids['DM'], :]
            actions_dmp1 = actions_p1[cond_ids['DM']]

            # get observation key and values for p1 p2
            o_keys = np.zeros((n_trials, T_total))
            o_vals = np.zeros((n_trials, T_total))
            for i in trial_id:
                o_keys[i], _, o_vals[i] = get_oq_keys(X_raw[i], task)
            o_keys_p1, o_keys_p2 = o_keys[:, :T_part], o_keys[:, T_part:]
            o_vals_p1, o_vals_p2 = o_vals[:, :T_part], o_vals[:, T_part:]
            o_keys_dmp1 = o_keys_p1[cond_ids['DM']]
            o_keys_dmp2 = o_keys_p2[cond_ids['DM']]
            o_vals_dmp1 = o_vals_p1[cond_ids['DM']]
            o_vals_dmp2 = o_vals_p2[cond_ids['DM']]

            '''plotting params'''
            alpha = .5
            n_se = 3
            # colors
            gr_pal = sns.color_palette('colorblind')[2:4]
            # make dir to save figs
            fig_dir = os.path.join(log_subpath['figs'], test_data_subdir)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            '''Schematicity influence'''

            schema_consistency = np.array([
                np.sum(np.argmax(def_path, axis=1) == targets_i[T_part:])
                for targets_i in targets
            ])
            schema_consistency -= np.min(schema_consistency)
            schema_consistency = schema_consistency / \
                np.max(schema_consistency)
            # plt.hist(schema_consistency)
            np.shape(corrects)
            cond_ = 'DM'
            dvs = [corrects, dks, mistakes]
            dv_names = ['corrects', 'dks', 'mistakes']
            f, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i, dv_i in enumerate(dvs):
                dv = np.mean(dv_i[:, T_part:], axis=1)
                r_val, p_val = pearsonr(
                    schema_consistency[cond_ids[cond_]], dv[cond_ids[cond_]]
                )
                sns.regplot(
                    schema_consistency[cond_ids[cond_]], dv[cond_ids[cond_]],
                    scatter_kws={'s': 40, 'alpha': .5},
                    x_jitter=.1, y_jitter=.01,
                    ax=axes[i],
                )
                axes[i].set_title('r = %.2f, p = %.2f' % (r_val, p_val))
                axes[i].set_ylabel(dv_names[i])
                axes[i].set_xlabel('Schematicity')
                axes[i].set_xlim([0, 1])
            sns.despine()
            f.tight_layout()
            fig_path = os.path.join(
                fig_dir, f'performance-{cond_}-schema-effect.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''plot behavioral performance - vertically aligned'''
            input_dict = {'Y': Y, 'dist_a': dist_a, 'cond_ids': cond_ids}
            pickle_save_dict(input_dict, f'temp/enc{enc_size_test}.pkl')

            # f, axes = plt.subplots(3, 1, figsize=(7, 9))

            for i, cn in enumerate(all_conds):
                # f, ax = plt.subplots(1, 1, figsize=(7, 3.5))
                Y_ = Y[cond_ids[cn], :]
                dist_a_ = dist_a[cond_ids[cn], :]
                # compute performance for this condition
                acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
                dk_mu = compute_dk(dist_a_)
                mis_mu, mis_er = compute_mistake(Y_, dist_a_, return_er=True)

                # cache data for all cond-subj
                acc_dict[cn]['mu'][i_s] = acc_mu
                acc_dict[cn]['er'][i_s] = acc_er
                mis_dict[cn]['mu'][i_s] = mis_mu
                mis_dict[cn]['er'][i_s] = mis_er
                dk_dict[cn]['mu'][i_s] = dk_mu

            '''P(error | schema cons) vs. P(error | schema in-cons)'''
            if n_def_tps > 0:
                f, axes = plt.subplots(1, 3, figsize=(17, 5))
                for i, cn in enumerate(all_conds):
                    mis_c = mis_dict[cn]['mu'][i_s][T_part:][np.array(
                        def_tps).astype(np.bool)]
                    mis_ic = mis_dict[cn]['mu'][i_s][T_part:][~np.array(
                        def_tps).astype(np.bool)]
                    mis_er_c = mis_dict[cn]['er'][i_s][T_part:][np.array(
                        def_tps).astype(np.bool)]
                    mis_er_ic = mis_dict[cn]['er'][i_s][T_part:][np.array(
                        def_tps).astype(np.bool)]

                    heights = [np.mean(mis_c), np.mean(mis_ic)]
                    yerrs = [np.std(mis_er_c), np.std(mis_er_ic)]
                    xticklabels = ['schematic', 'nonschematic']
                    xticks = range(len(heights))

                    # f, ax = plt.subplots(1, 1, figsize=(6, 5))
                    axes[i].bar(
                        x=xticks, height=heights, yerr=yerrs,
                        color=sns.color_palette('colorblind')[3]
                    )
                    axes[i].axhline(0, color='grey', linestyle='--')
                    axes[i].set_title(cn)
                    axes[i].set_xlabel('Transition type')
                    axes[i].set_xticks(xticks)
                    axes[i].set_ylim([-.05, .5])
                    # axes[i].set_ylim([-.015, .15])
                    axes[i].set_xticklabels(xticklabels)
                    axes[i].set_ylabel('P(error)')
                    f.tight_layout()
                    sns.despine()
                fig_path = os.path.join(fig_dir, f'error-schema-effect.png')
                f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''plot behavioral performance - each cond separately, 2nd part'''

            f, axes = plt.subplots(1, 3, figsize=(12, 4))
            for i, cn in enumerate(['RM', 'DM', 'NM']):
                Y_ = Y[cond_ids[cn], :]
                dist_a_ = dist_a[cond_ids[cn], :]
                # compute performance for this condition
                acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
                dk_mu = compute_dk(dist_a_)

                if i == 0:
                    add_legend = True
                    show_ylabel = True
                    legend_loc = (.33, .7)

                else:
                    add_legend = False
                    show_ylabel = False

                # plot
                plot_pred_acc_rcl(
                    acc_mu[T_part:], acc_er[T_part:],
                    acc_mu[T_part:] + dk_mu[T_part:],
                    p, f, axes[i],
                    title=f'{cn}',
                    add_legend=add_legend, legend_loc=legend_loc,
                    show_ylabel=show_ylabel
                )
                # axes[i].set_ylabel()
                axes[i].set_ylim([-.05, 1.05])
            fig_path = os.path.join(fig_dir, f'tz-acc-horizontal.png')
            f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''schema effect analysis
            '''
            def_path_int = np.argmax(def_path, axis=1)
            n_trials_dm = np.sum(cond_ids['DM'])
            time_hasproto = np.array(def_tps).astype(np.bool)

            # proto_response = np.argmax(def_path, axis=1)
            # proto_response_dmp2 = np.tile(
            #     proto_response, (np.sum(cond_ids['DM']), 1))
            #
            # schema_consistent_responses = actions_dmp2 == proto_response_dmp2
            # schema_consistent_responses = np.logical_and(
            #     time_hasproto, schema_consistent_responses)
            #
            # dp_consistent = targets_dmp2 == np.tile(
            #     def_path_int, [n_trials_dm, 1])
            #
            # dp_consistent_wproto = dp_consistent[:, time_hasproto]
            # dp_consistent_woproto = dp_consistent[:, ~time_hasproto]
            #
            # corrects_dmp2_wproto = corrects_dmp2[:, time_hasproto]
            # corrects_dmp2_woproto = corrects_dmp2[:, ~time_hasproto]
            # mistakes_dmp2_wproto = mistakes_dmp2[:, time_hasproto]
            # mistakes_dmp2_woproto = mistakes_dmp2[:, ~time_hasproto]
            # dks_dmp2_wproto = dks_dmp2[:, time_hasproto]
            # dks_dmp2_woproto = dks_dmp2[:, ~time_hasproto]
            #
            # n_mistakes = np.sum(mistakes_dmp2_wproto)
            # n_dks = np.sum(dks_dmp2_wproto)
            # n_corrects = np.sum(corrects_dmp2_wproto)
            #
            # # np.shape(schema_consistent_responses[:, time_hasproto])
            # sc_responses_wproto = schema_consistent_responses[:, time_hasproto]
            # sc_mistakes = np.logical_and(
            #     sc_responses_wproto, mistakes_dmp2_wproto)
            # n_sc_mistakes = np.sum(sc_mistakes)
            # n_sic_mistakes = n_mistakes - np.sum(n_sc_mistakes)
            #
            # corrects_dmp2_wproto_c = corrects_dmp2_wproto[dp_consistent_wproto]
            # corrects_dmp2_wproto_ic = corrects_dmp2_wproto[~dp_consistent_wproto]
            # corrects_dmp2_woproto_c = corrects_dmp2_woproto[dp_consistent_woproto]
            # corrects_dmp2_woproto_ic = corrects_dmp2_woproto[~dp_consistent_woproto]
            #
            # mistakes_dmp2_wproto_c = mistakes_dmp2_wproto[dp_consistent_wproto]
            # mistakes_dmp2_wproto_ic = mistakes_dmp2_wproto[~dp_consistent_wproto]
            # mistakes_dmp2_woproto_c = mistakes_dmp2_woproto[dp_consistent_woproto]
            # mistakes_dmp2_woproto_ic = mistakes_dmp2_woproto[~dp_consistent_woproto]
            #
            # dks_dmp2_wproto_c = dks_dmp2_wproto[dp_consistent_wproto]
            # dks_dmp2_wproto_ic = dks_dmp2_wproto[~dp_consistent_wproto]
            # dks_dmp2_woproto_c = dks_dmp2_woproto[dp_consistent_woproto]
            # dks_dmp2_woproto_ic = dks_dmp2_woproto[~dp_consistent_woproto]
            #
            # inpt_wproto = inpt_dmp2[:, time_hasproto]
            # inpt_woproto = inpt_dmp2[:, ~time_hasproto]
            #
            # inpt_wproto_c = inpt_wproto[dp_consistent_wproto]
            # inpt_wproto_ic = inpt_wproto[~dp_consistent_wproto]
            # inpt_woproto_c = inpt_woproto[dp_consistent_woproto]
            # inpt_woproto_ic = inpt_woproto[~dp_consistent_woproto]
            #
            # inpt_wproto_c_g[i_s] = inpt_wproto_c
            # inpt_wproto_ic_g[i_s] = inpt_wproto_ic
            # inpt_woproto_c_g[i_s] = inpt_woproto_c
            # inpt_woproto_ic_g[i_s] = inpt_woproto_ic
            # corrects_dmp2_wproto_c_g[i_s] = corrects_dmp2_wproto_c
            # corrects_dmp2_wproto_ic_g[i_s] = corrects_dmp2_wproto_ic
            # corrects_dmp2_woproto_c_g[i_s] = corrects_dmp2_woproto_c
            # corrects_dmp2_woproto_ic_g[i_s] = corrects_dmp2_woproto_ic
            #
            # dk_wproto_cic = [np.mean(dks_dmp2_wproto_c),
            #                  np.mean(dks_dmp2_wproto_ic)]
            # dk_woproto_cic = [
            #     np.mean(dks_dmp2_woproto_c), np.mean(dks_dmp2_woproto_ic)]
            #
            # inpt_wproto_cic = [np.mean(inpt_wproto_c), np.mean(inpt_wproto_ic)]
            # inpt_woproto_cic = [np.mean(inpt_woproto_c),
            #                     np.mean(inpt_woproto_ic)]
            # xticks = range(len(heights))
            # xticklabels = ['consistent', 'violated']
            # f, axes = plt.subplots(1, 2, figsize=(12, 5))
            #
            # axes[0].bar(
            #     x=xticks, height=inpt_wproto_cic,
            # )
            # axes[0].set_xlabel('Has a prototypical event')
            #
            # axes[1].bar(
            #     x=xticks, height=inpt_woproto_cic,
            # )
            # axes[1].set_xlabel('Has no prototypical event')
            # for ax in axes:
            #     ax.axhline(0, color='grey', linestyle='--')
            #     ax.set_xticks(xticks)
            #     ax.set_xticklabels(xticklabels)
            #     ax.set_ylabel('Input gate')
            # f.tight_layout()
            # sns.despine()
            # fig_path = os.path.join(fig_dir, f'schema-input.png')
            # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
            #
            # xticks = range(len(heights))
            # xticklabels = ['consistent', 'violated']
            # f, axes = plt.subplots(1, 2, figsize=(12, 5))
            # corrects_dmp2_wproto_cic = [np.mean(corrects_dmp2_wproto_c),
            #                             np.mean(corrects_dmp2_wproto_ic)]
            # corrects_dmp2_woproto_cic = [np.mean(corrects_dmp2_woproto_c),
            #                              np.mean(corrects_dmp2_woproto_ic)]
            #
            # axes[0].bar(x=xticks, height=corrects_dmp2_wproto_cic)
            # axes[0].set_xlabel('Has a prototypical event')
            #
            # axes[1].bar(x=xticks, height=corrects_dmp2_woproto_cic)
            # axes[1].set_xlabel('Has no prototypical event')
            # for ax in axes:
            #     ax.axhline(0, color='grey', linestyle='--')
            #     ax.set_xticks(xticks)
            #     ax.set_xticklabels(xticklabels)
            #     ax.set_ylabel('Correct rate')
            # f.tight_layout()
            # sns.despine()
            # fig_path = os.path.join(fig_dir, f'schema-correct.png')
            # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
            #
            # # performance as a func of input gate values
            # f, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
            # sns.regplot(inpt_wproto_ic, corrects_dmp2_wproto_ic.astype(
            #     np.float32), logistic=True, ax=axes[0])
            # # ax.set_xlabel('Input gate')
            # axes[0].set_ylabel('Correct')
            # sns.regplot(inpt_wproto_ic, mistakes_dmp2_wproto_ic.astype(
            #     np.float32), logistic=True, ax=axes[1])
            # # ax.set_xlabel('Input gate')
            # axes[1].set_ylabel('Mistake')
            # sns.regplot(inpt_wproto_ic, mistakes_dmp2_wproto_ic.astype(
            #     np.float32), logistic=True, ax=axes[2])
            # axes[2].set_ylabel('Don\'t know')
            # axes[-1].set_xlabel('Input gate')
            #
            # f.tight_layout()
            # sns.despine()
            # fig_path = os.path.join(fig_dir, f'schema-input-behav.png')
            # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
            #
            # corrects_dmp2_wwoproto_cic_g[i_s] = np.array([
            #     corrects_dmp2_wproto_cic,
            #     corrects_dmp2_woproto_cic
            # ])
            # inpt_wwoproto_cic_g[i_s] = np.array([
            #     inpt_wproto_cic,
            #     inpt_woproto_cic
            # ])
            # dk_wwoproto_cic_g[i_s] = np.array([
            #     dk_wproto_cic, dk_woproto_cic
            # ])
            # # dk_wproto_cic, dk_woproto_cic
            #
            # n_dks_g[i_s] = n_dks
            # n_corrects_g[i_s] = n_corrects
            # n_sc_mistakes_g[i_s] = n_sc_mistakes
            # n_sic_mistakes_g[i_s] = n_sic_mistakes
            #
            # schema_grid_data = {
            #     'corrects': corrects_dmp2_wwoproto_cic_g,
            #     'dks': dk_wwoproto_cic_g,
            #     'inpt': inpt_wwoproto_cic_g,
            #     'n_sc_mistakes': n_sc_mistakes_g,
            #     'n_sic_mistakes': n_sic_mistakes_g,
            #     'n_dks': n_dks_g,
            #     'n_corrects': n_corrects_g
            # }
            # pickle_save_dict(schema_grid_data, 'temp/schema-%.2f' % def_prob)

            '''
            '''
            # dk_norms = np.linalg.norm(DA_p2[dks_p2], axis=1)
            # ndk_norms = np.linalg.norm(DA_p2[~dks_p2], axis=1)
            # dk_norm_mu, dk_norm_se = compute_stats(dk_norms)
            # ndk_norm_mu, ndk_norm_se = compute_stats(ndk_norms)
            #
            # f, ax = plt.subplots(1, 1, figsize=(5, 5))
            # xticks = range(2)
            # ax.bar(x=xticks, height=[dk_norm_mu, ndk_norm_mu],
            #        yerr=np.array([dk_norm_se, ndk_norm_se]) * 3)
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(['uncertain', 'certain'])
            # ax.set_ylabel('Activity norm')
            # f.tight_layout()
            # sns.despine()
            #
            # proto_response_p2 = np.tile(
            #     proto_response, (np.shape(actions)[0], 1))
            # schema_consistent_responses_p2 = actions[:,
            #                                          T_part:] == proto_response_p2
            # schema_inconsistent_responses_p2 = np.logical_and(
            #     ~dks_p2, ~schema_consistent_responses_p2)
            #
            # dk_norms = np.linalg.norm(DA_p2[dks_p2], axis=1)
            # sc_norms = np.linalg.norm(
            #     DA_p2[schema_consistent_responses_p2], axis=1)
            # sic_norms = np.linalg.norm(
            #     DA_p2[schema_inconsistent_responses_p2], axis=1)
            #
            # dk_norm_mu, dk_norm_se = compute_stats(dk_norms)
            # sc_norm_mu, sc_norm_se = compute_stats(sc_norms)
            # sic_norm_mu, sic_norm_se = compute_stats(sic_norms)
            #
            # f, ax = plt.subplots(1, 1, figsize=(7, 5))
            # xticks = range(3)
            # ax.bar(x=xticks, height=[dk_norm_mu, sc_norm_mu, sic_norm_mu],
            #        yerr=np.array([dk_norm_se, sc_norm_se, sic_norm_se]) * 3)
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(
            #     ['uncertain', 'schema\nconsistent', 'schema\ninconsistent'])
            # ax.set_ylabel('Activity norm')
            # f.tight_layout()
            # sns.despine()

            '''decoding data-prep
            '''
            # from sklearn.datasets import load_breast_cancer
            from sklearn.linear_model import RidgeClassifier, LogisticRegression
            from sklearn.model_selection import PredefinedSplit
            # from sklearn.svm import LinearSVC

            def build_yob(o_keys_p, o_vals_p, def_yob_val=-1):
                Yob_p = np.full((n_trials, T_part, T_part), def_yob_val)
                for i in range(n_trials):
                    # construct Y for the t-th classifier
                    for t in range(T_part):
                        time_observed = np.argmax(o_keys_p[i] == t)
                        # the y for the i-th trial for the t-th feature
                        y_it = np.full((T_part), def_yob_val)
                        y_it[time_observed:] = o_vals_p[i][time_observed]
                        Yob_p[i, :, t] = y_it
                return Yob_p

            # reformat X
            CM_p1rs = np.reshape(CM_p1, (n_trials * T_part, -1))
            DA_p1rs = np.reshape(DA_p1, (n_trials * T_part, -1))
            CM_p2rs = np.reshape(CM_p2, (n_trials * T_part, -1))
            DA_p2rs = np.reshape(DA_p2, (n_trials * T_part, -1))

            # build y
            Yob_p1 = build_yob(o_keys_p1, o_vals_p1)
            Yob_p2 = build_yob(o_keys_p2, o_vals_p2)

            # precompute mistakes-related variables
            n_mistakes_per_trial = np.sum(mistakes_dmp2, axis=1)
            has_mistake = n_mistakes_per_trial > 0

            # split trials w/ vs. w/o mistakes
            actions_dmp1hm = actions_dmp1[has_mistake, :]
            targets_dmp1hm = targets_dmp1[has_mistake, :]
            actions_dmp2hm = actions_dmp2[has_mistake, :]
            targets_dmp2hm = targets_dmp2[has_mistake, :]
            corrects_dmp2hm = corrects_dmp2[has_mistake, :]
            mistakes_dmp2hm = mistakes_dmp2[has_mistake, :]
            dks_dmp2hm = dks_dmp2[has_mistake, :]
            CM_dmp2hm = CM_dmp2[has_mistake, :, :]
            DA_dmp2hm = DA_dmp2[has_mistake, :, :]
            o_keys_dmp1hm = o_keys_dmp1[has_mistake, :]
            o_keys_dmp2hm = o_keys_dmp2[has_mistake, :]
            o_vals_dmp1hm = o_vals_dmp1[has_mistake, :]
            o_vals_dmp2hm = o_vals_dmp2[has_mistake, :]

            actions_dmp1nm = actions_dmp1[~has_mistake, :]
            targets_dmp1nm = targets_dmp1[~has_mistake, :]
            actions_dmp2nm = actions_dmp2[~has_mistake, :]
            targets_dmp2nm = targets_dmp2[~has_mistake, :]
            corrects_dmp2nm = corrects_dmp2[~has_mistake, :]
            mistakes_dmp2nm = mistakes_dmp2[~has_mistake, :]
            dks_dmp2nm = dks_dmp2[~has_mistake, :]
            CM_dmp2nm = CM_dmp2[~has_mistake, :, :]
            DA_dmp2nm = DA_dmp2[~has_mistake, :, :]
            o_keys_dmp1nm = o_keys_dmp1[~has_mistake, :]
            o_keys_dmp2nm = o_keys_dmp2[~has_mistake, :]
            o_vals_dmp1nm = o_vals_dmp1[~has_mistake, :]
            o_vals_dmp2nm = o_vals_dmp2[~has_mistake, :]

            o_keys_dmhm = np.hstack([o_keys_dmp1hm, o_keys_dmp2hm])
            o_vals_dmhm = np.hstack([o_vals_dmp1hm, o_vals_dmp2hm])
            actions_dmhm = np.hstack([actions_dmp1hm, actions_dmp2hm])
            targets_dmhm = np.hstack([targets_dmp1hm, targets_dmp2hm])

            o_keys_dmnm = np.hstack([o_keys_dmp1nm, o_keys_dmp2nm])
            o_vals_dmnm = np.hstack([o_vals_dmp1nm, o_vals_dmp2nm])
            actions_dmnm = np.hstack([actions_dmp1nm, actions_dmp2nm])
            targets_dmnm = np.hstack([targets_dmp1nm, targets_dmp2nm])

            actions_dmnm[actions_dmnm == n_branch] = -1
            actions_dmhm[actions_dmhm == n_branch] = -1

            actions_dmhm += 1
            actions_dmnm += 1
            targets_dmhm += 1
            targets_dmnm += 1

            '''decoding - train on ground truth feature presence during part 1'''

            # # build trial id matrix
            # trial_id_mat = np.tile(trial_id, (T_part, 1)).T
            # trial_id_unroll = np.reshape(trial_id_mat, (n_trials * T_part, ))
            #
            # rc_alpha = 1
            # cm_rc = [LogisticRegression(penalty='l2', C=rc_alpha)
            #          for _ in range(T_part)]
            #
            # cm_scores = np.zeros((n_trials, T_part))
            # Yobrs_p1_hat = np.zeros((n_trials, T_part, T_part))
            # Yobrs_p1_proba = np.zeros((n_trials, T_part, T_part, n_branch + 1))
            # for t in range(T_part):
            #     Yobrs_t = np.reshape(Yob_p1[:, :, t], (n_trials * T_part, -1))
            #
            #     for i in trial_id:
            #         mask = trial_id_unroll == i
            #
            #         Yobrs_t_tr = Yobrs_t[~mask]
            #         Yobrs_t_te = Yobrs_t[mask]
            #         CM_p1rs_tr = CM_p1rs[~mask, :]
            #         CM_p1rs_te = CM_p1rs[mask, :]
            #
            #         cm_rc[t].fit(CM_p1rs_tr, Yobrs_t_tr)
            #         cm_scores[i, t] = cm_rc[t].score(CM_p1rs_te, Yobrs_t_te)
            #         Yobrs_p1_hat[i, t, :] = cm_rc[t].predict(CM_p1rs_te)
            #         Yobrs_p1_proba[i, t, :] = cm_rc[t].predict_proba(
            #             CM_p1rs_te)
            #
            # print(cm_scores.mean())
            #
            # f, ax = plt.subplots(1, 1, figsize=(7, 4))
            # cm_scores_mu = np.mean(cm_scores, axis=0)
            # ax.plot(cm_scores_mu)
            # ax.set_xlabel('Time')
            # ax.set_ylabel('Decoding accuracy')
            # sns.despine()
            # f.tight_layout()
            #
            # Yobrs_p2_hat_sr = np.zeros((n_trials * T_part, T_part))
            # Yobrs_p2_proba_sr = np.zeros(
            #     (n_trials * T_part, T_part, n_branch + 1))
            # for t in range(T_part):
            #     Yobrs_t = np.reshape(Yob_p1[:, :, t], (n_trials * T_part, -1))
            #     cm_rc[t].fit(CM_p1rs, Yobrs_t)
            #     Yobrs_p2_hat_sr[:,  t] = cm_rc[t].predict(CM_p2rs)
            #     Yobrs_p2_proba_sr[:, t] = cm_rc[t].predict_proba(CM_p2rs)
            #     # da_rc[t].fit(DA_p1rs, Yobrs_t)
            #     # Yobrs_p2_hat_sr[:,  t] = da_rc[t].predict(DA_p2rs)
            # Yobrs_p2_hat = Yobrs_p2_hat_sr.reshape((n_trials, T_part, T_part))
            # Yobrs_p2_proba = Yobrs_p2_proba_sr.reshape(
            #     (n_trials, T_part, T_part, n_branch + 1))
            # # np.shape(Yobrs_p2_proba)
            #
            # def round_predictions(predictions):
            #     predictions_round = np.round(predictions)
            #     predictions_round[predictions_round >= 0] = 1
            #     predictions_round[predictions_round < 0] = 0
            #     return predictions_round
            #
            # Yobrs_p2_hat_round = round_predictions(Yobrs_p2_hat)
            # Yobrs_p2_hat_rm = Yobrs_p2_hat[cond_ids['RM']]
            # Yobrs_p2_hat_dm = Yobrs_p2_hat[cond_ids['DM']]
            # Yobrs_p2_hat_nm = Yobrs_p2_hat[cond_ids['NM']]
            #
            # plt.imshow(Yobrs_p2_hat[cond_ids['DM']][1, :, :])
            #
            # Yobrs_p2_hat_round[cond_ids['DM']]
            # Yp2hatrdm = np.transpose(
            #     Yobrs_p2_hat_round[cond_ids['DM']], (0, 2, 1))
            # Yp2hatrdm_wp = Yp2hatrdm[:, time_hasproto, :]
            # Yp2hatrdm_wop = Yp2hatrdm[:, ~time_hasproto, :]
            #
            # Yobrs_dmp1_proba = Yobrs_p1_proba[cond_ids['DM']]
            # Yobrs_dmp2_proba = Yobrs_p2_proba[cond_ids['DM']]
            #
            # np.shape(Yobrs_p2_hat_rm)
            # plt.imshow(Yobrs_p2_hat_rm[10, :, :], aspect='auto')
            # plt.imshow(np.mean(Yobrs_p2_hat_rm, axis=0), aspect='auto')
            # plt.imshow(np.mean(Yobrs_p2_hat_round, axis=0), aspect='auto')
            #
            # plt.imshow(Yobrs_p2_hat_dm[0, :, :], aspect='auto')
            # plt.imshow(np.mean(Yobrs_p2_hat_dm, axis=0), aspect='auto')
            #
            # plt.imshow(Yobrs_p2_hat_nm[0, :, :], aspect='auto')
            # plt.imshow(np.mean(Yobrs_p2_hat_nm, axis=0), aspect='auto')
            #
            # # plt.imshow(np.argmax(Y[0], axis=1))
            # ydec_rm_mu, ydec_rm_se = compute_stats(
            #     np.sum(Yobrs_p2_hat_round[cond_ids['RM']], axis=2))
            # ydec_dm_mu, ydec_dm_se = compute_stats(
            #     np.sum(Yobrs_p2_hat_round[cond_ids['DM']], axis=2))
            # ydec_nm_mu, ydec_nm_se = compute_stats(
            #     np.sum(Yobrs_p2_hat_round[cond_ids['NM']], axis=2))
            #
            # # # plot part 1 decoding results
            # # Yobrs_p1_hat_t = np.transpose(Yobrs_p1_hat, (0, 2, 1))
            # # Yobrs_p1_hat_round = round_predictions(Yobrs_p1_hat_t)
            # # ydec_rm_mu, ydec_rm_se = compute_stats(
            # #     np.sum(Yobrs_p1_hat_round[cond_ids['RM']], axis=2))
            # # ydec_dm_mu, ydec_dm_se = compute_stats(
            # #     np.sum(Yobrs_p1_hat_round[cond_ids['DM']], axis=2))
            # # ydec_nm_mu, ydec_nm_se = compute_stats(
            # #     np.sum(Yobrs_p1_hat_round[cond_ids['NM']], axis=2))
            #
            # # plot #decodable features
            # f, ax = plt.subplots(1, 1, figsize=(7, 5))
            # ax.errorbar(range(T_part), y=ydec_rm_mu, yerr=ydec_rm_se)
            # ax.errorbar(range(T_part), y=ydec_dm_mu, yerr=ydec_dm_se)
            # ax.errorbar(range(T_part), y=ydec_nm_mu, yerr=ydec_nm_se)
            #
            # ax.legend(['RM', 'DM', 'NM'])
            # ax.set_xlabel('Time, part 2')
            # ax.set_ylabel('# decodable features')
            # f.tight_layout()
            # sns.despine()
            # fig_path = os.path.join(fig_dir, f'mvpa-n-features.png')
            # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''single trial analysis
            for each dm trial, for every time point during part two
            '''
            # def feat_vec2mat(decoded_feat_vec):
            #     decoded_feat_mat = np.zeros((n_branch, T_total))
            #     for v in range(n_branch):
            #         decoded_feat_mat[v, :] = np.array(
            #             decoded_feat_vec == v).astype(int)
            #     return decoded_feat_mat
            #
            # mistakes_dmp2
            # Yobrs_p2_hat = np.transpose(Yobrs_p2_hat, (0, 2, 1))
            # Yobrs_hat = np.concatenate([Yobrs_p1_hat, Yobrs_p2_hat], axis=2)
            # Yobrs_hat_dm = Yobrs_hat[cond_ids['DM'], :, :]
            # np.shape(Yobrs_hat)
            #
            # plt.imshow(Yobrs_hat_dm[3])
            #
            # # for each trial
            # for i in len(Yobrs_hat_dm):
            #     for t in range(T_total):
            #         decoded_feat_mat_it = feat_vec2mat(Yobrs_hat_dm[i, t])
            #         Yobrs_hat_dm[i, t] == 1

            '''decoding - train on ground truth feature presence during part 1,
            NM/RM during part 2, and inferred feature presence during DM part 2
            '''
            # data prep
            CM = np.hstack([CM_p1, CM_p2])
            Yob = np.hstack([Yob_p1, Yob_p2])
            CM_rs = np.reshape(CM, (n_trials * T_total, n_hidden))
            Yob_rs = np.reshape(Yob, (n_trials * T_total, T_part))

            # for the RM condition
            # assert no feature is unobserved by the end of part 1
            assert np.sum(Yob[cond_ids['RM']][:, T_part - 1, :] == -1) == 0
            # fill the part 2 to be observed
            for t in np.arange(T_part, T_total):
                Yob[cond_ids['RM'], t, :] = Yob[cond_ids['RM'], T_part - 1, :]
            # for the DM condition
            # estimated recall time
            rt_est = np.argmax(inpt_p2, axis=1)
            dm_ids = np.where(cond_ids['DM'])[0]
            # for the ith DM trial
            for i in range(np.sum(cond_ids['DM'])):
                rti = rt_est[i]
                ii = dm_ids[i]
                # fill the part 2, after recall, to be observed
                for t in np.arange(T_part + rti, T_total):
                    Yob[ii, t, :] = Yob[ii, T_part - 1, :]

            # build trial id matrix
            trial_id_mat = np.tile(trial_id, (T_total, 1)).T
            trial_id_unroll = np.reshape(trial_id_mat, (n_trials * T_total, ))

            # set up holdout set params
            n_trials_factors = find_factors(n_trials)
            if len(n_trials_factors) == 2:
                n_test_trial = n_trials_factors[0]
            else:
                m_ = np.max(np.where(np.array(n_trials_factors) < 10)[0])
                n_test_trial = n_trials_factors[m_]

            # n_test_trial = n_trials_factors[2]
            n_test_tps = n_test_trial * T_total
            testset_ids = np.reshape(trial_id, (-1, n_test_trial))
            n_folds = np.shape(testset_ids)[0]

            # start decoding
            rc_alpha = 1
            cm_rc = [LogisticRegression(penalty='l2', C=rc_alpha)
                     for _ in range(T_part)]
            Yob_hat_ = np.zeros((n_folds, n_test_tps, T_part))
            Yob_proba_ = np.zeros((n_trials, T_part, T_total, n_branch + 1))
            for fid, testset_ids_i in enumerate(testset_ids):
                tmask = np.logical_and(
                    trial_id_unroll >= testset_ids_i[0],
                    trial_id_unroll <= testset_ids_i[-1])
                # for the n/t-th classifier
                for n in range(T_part):
                    # print(n)
                    cm_rc[n].fit(CM_rs[~tmask], Yob_rs[~tmask, n])
                    Yob_hat_[fid, :, n] = cm_rc[n].predict(CM_rs[tmask])
                    # probabilistic estimates for the i-th
                    for ii in testset_ids_i:
                        Yob_proba_[ii, n] = cm_rc[n].predict_proba(CM[ii])

            Yob_hat_rs = np.reshape(Yob_hat_, np.shape(Yob_rs))
            Yob_hat = np.reshape(Yob_hat_rs, np.shape(Yob))
            acc = np.sum(Yob_hat == Yob) / Yob.size
            # print(acc)

            Yob_hat_p2 = Yob_hat[:, T_part:, :]
            Yob_hat_p2_v = np.vstack([Yob_hat_p2[:, t, t]
                                      for t in range(T_part)]).T
            # plt.imshow(Yob_hat_p2_v)
            # np.shape(np.vstack([Yob_hat_p2[:, t, t] for t in range(T_part)]).T)

            # np.shape(actions_p2)
            yconsistency = np.sum(Yob_hat_p2_v == actions_p2) / actions_p2.size
            print(yconsistency, acc)
            # plt.imshow(Yob_rs[~tmask, :], aspect='auto')

            '''plot'''
            # Yobrs_dmp1hm_proba = Yobrs_dmp1_proba[has_mistake, :]
            # Yobrs_dmp2hm_proba = Yobrs_dmp2_proba[has_mistake, :]
            # Yobrs_dmp1nm_proba = Yobrs_dmp1_proba[~has_mistake, :]
            # Yobrs_dmp2nm_proba = Yobrs_dmp2_proba[~has_mistake, :]

            # for the i-th mistakes trial, plot the j-th mistake
            i = 0
            j = 0
            for i in range(np.shape(mistakes_dmp2hm)[0]):
                # when/what feature were mistaken
                mistake_feature_i = np.where(mistakes_dmp2hm[i, :])[0]
                for j in range(len(mistake_feature_i)):

                    decoded_feat_mat = Yob_proba_[i, mistake_feature_i[j]]
                    # decoded_feat_mat=np.vstack([
                    #     Yobrs_dmp1hm_proba[i, mistake_feature_i[j], :, :],
                    #     Yobrs_dmp1hm_proba[i, mistake_feature_i[j], :, :]
                    # ])

                    feat_otimes = np.where(
                        o_keys_dmhm[i] == mistake_feature_i[j])[0]
                    feat_ovals = o_vals_dmhm[i][feat_otimes]

                    feat_qtimes = mistake_feature_i[j] + np.array([0, T_part])

                    f, ax = plt.subplots(1, 1, figsize=(9, 4))
                    # ax.imshow(decoded_feat_mat, aspect='auto', cmap='bone')
                    ax.imshow(
                        decoded_feat_mat.T, aspect='auto', cmap='bone')
                    ax.axvline(T_part - .5, linestyle='--', color='grey')
                    # ax.axvline(feat_qtimes[0], linestyle='--', color='orange')
                    # ax.axvline(feat_qtimes[1], linestyle='--', color='orange')

                    for fot, fqt in zip(feat_otimes, feat_qtimes):
                        rect = patches.Rectangle(
                            (fot - .5, targets_dmhm[i, :][fqt] - .5), 1, 1,
                            edgecolor='green', facecolor='none', linewidth=3
                        )

                        ax.add_patch(rect)
                    for fqt in feat_qtimes:
                        rect = patches.Rectangle(
                            (fqt - .5, actions_dmhm[i, :][fqt] - .5), 1, 1,
                            edgecolor='orange', facecolor='none', linewidth=3
                        )
                        ax.add_patch(rect)
                    if time_hasproto[feat_qtimes[0]]:
                        ax.scatter(
                            feat_qtimes, 1 + np.array([def_path_int[feat_qtimes[0]]] * 2), s=50, color='red')

                    ax.set_xlabel('Part 1                    Part 2')
                    ax.set_ylabel('Choice')
                    ax.set_xticks([0, T_part - 1, T_total - 1])
                    ax.set_xticklabels([0, T_part - 1, T_total - 1])
                    ax.set_yticks(np.arange(n_branch + 1))
                    ax.set_yticklabels(np.arange(n_branch + 1))
                    f.tight_layout()

                    td_dir_path = os.path.join(fig_dir, 'trial_data')
                    if not os.path.exists(td_dir_path):
                        os.makedirs(td_dir_path)

                    fig_path = os.path.join(td_dir_path, f'mistake-{i}-{j}')
                    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''corrects'''
            # corrects_dmp2nm
            i, j = 0, 0
            for i in range(np.shape(corrects_dmp2nm)[0]):
                # when/what feature were mistaken
                correct_feature_i = np.where(corrects_dmp2nm[i, :])[0]
                for j in range(len(correct_feature_i)):
                    decoded_feat_mat = Yob_proba_[i, correct_feature_i[j]]
                    # decoded_feat_mat = np.vstack([
                    #     Yobrs_dmp1nm_proba[i, correct_feature_i[j], :, :],
                    #     Yobrs_dmp2nm_proba[i, correct_feature_i[j], :, :]
                    # ])

                    feat_otimes = np.where(
                        o_keys_dmnm[i] == correct_feature_i[j])[0]
                    feat_ovals = o_vals_dmnm[i][feat_otimes]

                    feat_qtimes = correct_feature_i[j] + np.array([0, T_part])

                    f, ax = plt.subplots(1, 1, figsize=(9, 4))
                    # ax.imshow(decoded_feat_mat, aspect='auto', cmap='bone')
                    ax.imshow(
                        decoded_feat_mat.T, aspect='auto', cmap='bone')
                    ax.axvline(T_part - .5, linestyle='--', color='grey')
                    # ax.axvline(feat_qtimes[0], linestyle='--', color='orange')
                    # ax.axvline(feat_qtimes[1], linestyle='--', color='orange')

                    for fot, fqt in zip(feat_otimes, feat_qtimes):
                        rect = patches.Rectangle(
                            (fot - .5, targets_dmnm[i, :][fqt] - .5), 1, 1,
                            edgecolor='green', facecolor='none', linewidth=3
                        )
                        ax.add_patch(rect)
                    for fqt in feat_qtimes:
                        rect = patches.Rectangle(
                            (fqt - .5, actions_dmnm[i, :][fqt] - .5), 1, 1,
                            edgecolor='orange', facecolor='none', linewidth=3
                        )
                        ax.add_patch(rect)
                    if time_hasproto[feat_qtimes[0]]:
                        ax.scatter(
                            feat_qtimes, 1 + np.array([def_path_int[feat_qtimes[0]]] * 2), s=50, color='red')

                    ax.set_xlabel('Part 1                    Part 2')
                    ax.set_ylabel('Choice')
                    ax.set_xticks([0, T_part - 1, T_total - 1])
                    ax.set_xticklabels([0, T_part - 1, T_total - 1])
                    ax.set_yticks(np.arange(n_branch + 1))
                    ax.set_yticklabels(np.arange(n_branch + 1))
                    f.tight_layout()

                    td_dir_path = os.path.join(fig_dir, 'trial_data')
                    if not os.path.exists(td_dir_path):
                        os.makedirs(td_dir_path)

                    fig_path = os.path.join(td_dir_path, f'correct-{i}-{j}')
                    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

            '''recency at encoding -> error rate'''
        #     # xticks = np.arange(T_part)
        #     # error_counts_over_time = np.zeros(T_part)
        #     # for i in range(len(mistakes_dmp2hm)):
        #     #     tob = o_keys_dmp1hm[i][np.where(mistakes_dmp2hm[i])[0]]
        #     #     error_counts_over_time[tob.astype(int)] += 1
        #     # f, ax = plt.subplots(1, 1, figsize=(7, 5))
        #     # # time_hasproto
        #     # ax.bar(x=xticks, height=error_counts_over_time)
        #     # # ax.bar(x=xticks[time_hasproto],
        #     # #        height=error_counts_over_time[time_hasproto])
        #     # # ax.bar(x=xticks[~time_hasproto],
        #     # #        height=error_counts_over_time[~time_hasproto])
        #     # ax.set_xlabel('Time observed during part 1')
        #     # ax.set_ylabel('#errors during part 2')
        #     # ax.set_title('DM')
        #     # sns.despine()
        #     # f.tight_layout()
        #     # fig_path = os.path.join(fig_dir, f'err-enc-recency.png')
        #     # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''compute cell-memory similarity / memory activation '''
        #     # compute similarity between cell state vs. memories
        #     sim_cos, sim_lca = compute_cell_memory_similarity(
        #         C, V, inpt, leak, comp)
        #     sim_cos_dict = create_sim_dict(
        #         sim_cos, cond_ids, n_targ=p.n_segments)
        #     sim_lca_dict = create_sim_dict(
        #         sim_lca, cond_ids, n_targ=p.n_segments)
        #     sim_cos_stats = compute_cell_memory_similarity_stats(
        #         sim_cos_dict, cond_ids)
        #     sim_lca_stats = compute_cell_memory_similarity_stats(
        #         sim_lca_dict, cond_ids)
        #     ma_list[i_s] = sim_lca_stats
        #     ma_raw_list[i_s] = sim_lca_dict
        #     ma_cos_list[i_s] = sim_cos_stats
        #
        #     avg_ma = {cond: {m_type: None for m_type in memory_types}
        #               for cond in all_conds}
        #     for cond in all_conds:
        #         for m_type in memory_types:
        #             if sim_lca_dict[cond][m_type] is not None:
        #                 # print(np.shape(sim_lca_dict[cond][m_type]))
        #                 avg_ma[cond][m_type] = np.mean(
        #                     sim_lca_dict[cond][m_type], axis=-1
        #                 )
        #
        #     '''mem_act | schema cons vs. mem_act | schema in-cons'''
        #     if n_def_tps > 0:
        #         cn = 'DM'
        #
        #         ma_dm_p2 = avg_ma[cn]['targ'][:, T_part:]
        #         ma_dm_p2_c = ma_dm_p2[:, np.array(def_tps).astype(np.bool)]
        #         ma_dm_p2_ic = ma_dm_p2[:, ~np.array(def_tps).astype(np.bool)]
        #         ma_dm_p2_c = np.mean(ma_dm_p2_c, axis=1)
        #         ma_dm_p2_ic = np.mean(ma_dm_p2_ic, axis=1)
        #         heights = [np.mean(ma_dm_p2_c), np.mean(ma_dm_p2_ic)]
        #         xticks = range(len(heights))
        #         xticklabels = ['yes', 'no']
        #         f, ax = plt.subplots(1, 1, figsize=(6, 5))
        #         ax.bar(
        #             x=xticks, height=heights,
        #             color=sns.color_palette('colorblind')[2]
        #         )
        #         ax.axhline(0, color='grey', linestyle='--')
        #         ax.set_title(cn)
        #         ax.set_xlabel('Has a prototypical event?')
        #         ax.set_xticks(xticks)
        #         # ax.set_ylim([-.05, .5])
        #         ax.set_xticklabels(xticklabels)
        #         ax.set_ylabel('Memory activation')
        #         f.tight_layout()
        #         sns.despine()
        #
        #     '''plot target/lure activation for all conditions - horizontal'''
        #     # sim_stats_plt = {'LCA': sim_lca_stats, 'cosine': sim_cos_stats}
        #     ylim_bonds = {'LCA': None, 'cosine': None}
        #     ker_name, sim_stats_plt_ = 'LCA', sim_lca_stats
        #     # print(ker_name, sim_stats_plt_)
        #     tsf = (T_part + pad_len_test) / T_part
        #     f, axes = plt.subplots(1, 3, figsize=(12, 4))
        #     for i, c_name in enumerate(cond_ids.keys()):
        #         for m_type in memory_types:
        #             if m_type == 'targ' and c_name == 'NM':
        #                 continue
        #             color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
        #             axes[i].errorbar(
        #                 x=range(T_part),
        #                 y=sim_stats_plt_[c_name][m_type]['mu'][T_part:],
        #                 yerr=sim_stats_plt_[c_name][m_type]['er'][T_part:],
        #                 label=f'{m_type}', color=color_
        #             )
        #             axes[i].set_title(c_name)
        #             axes[i].set_xlabel('Time')
        #     axes[0].set_ylabel('Memory activation')
        #     axes[0].legend()
        #
        #     # make all ylims the same
        #     ylim_bonds[ker_name] = get_ylim_bonds(axes)
        #     ylim_bonds[ker_name] = (
        #         np.max((ylim_bonds[ker_name][0], -.05)
        #                ), np.round((ylim_bonds[ker_name][1] + .1), decimals=1)
        #     )
        #     # np.round(ylim_bonds[ker_name][1] + .1, decimals=1)
        #     # ylim_bonds[ker_name] = [-.05, .6]
        #     for i, ax in enumerate(axes):
        #         ax.set_ylim(ylim_bonds[ker_name])
        #         ax.set_xticks([0, p.env.n_param - 1])
        #         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #         ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #
        #     if pad_len_test > 0:
        #         for ax in axes:
        #             ax.axvline(pad_len_test, color='grey', linestyle='--')
        #     f.tight_layout()
        #     sns.despine()
        #     fig_path = os.path.join(fig_dir, f'tz-memact-{ker_name}-hori.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     # '''plot target/lure activation for all conditions'''
        #     # sim_stats_plt = {'LCA': sim_lca_stats, 'cosine': sim_cos_stats}
        #     # for ker_name, sim_stats_plt_ in sim_stats_plt.items():
        #     #     # print(ker_name, sim_stats_plt_)
        #     #     tsf = (T_part + pad_len_test) / T_part
        #     #     for i, c_name in enumerate(cond_ids.keys()):
        #     #         f, ax = plt.subplots(1, 1, figsize=(7 * tsf, 3.5))
        #     #         for m_type in memory_types:
        #     #             if m_type == 'targ' and c_name == 'NM':
        #     #                 continue
        #     #             color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
        #     #             ax.errorbar(
        #     #                 x=range(T_part),
        #     #                 y=sim_stats_plt_[c_name][m_type]['mu'][T_part:],
        #     #                 yerr=sim_stats_plt_[c_name][m_type]['er'][T_part:],
        #     #                 label=f'{m_type}', color=color_
        #     #             )
        #     #         ax.set_title(c_name)
        #     #         ax.set_ylabel('Memory activation')
        #     #         ax.set_xlabel('Time (part 2)')
        #     #         ax.legend()
        #     #
        #     #         # ax.set_ylim([-.05, .625])
        #     #         ax.set_ylim(ylim_bonds[ker_name])
        #     #         ax.set_xticks(np.arange(0, p.env.n_param, 5))
        #     #         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #     #         ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #     #
        #     #         if pad_len_test > 0:
        #     #             ax.axvline(pad_len_test, color='grey', linestyle='--')
        #     #         f.tight_layout()
        #     #         sns.despine()
        #     #         fig_path = os.path.join(
        #     #             fig_dir, f'tz-memact-{ker_name}-{c_name}.png')
        #     #         f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''correlate lure recall vs. behavior '''
        #     for cond_ in ['NM', 'DM']:
        #         lure_act_cp2 = sim_lca_dict[cond_]['lure'][:, T_part:, :]
        #         corrects_cp2 = corrects[cond_ids[cond_]][:, T_part:]
        #         mistakes_cp2 = mistakes[cond_ids[cond_]][:, T_part:]
        #         dks_cp2 = dks[cond_ids[cond_]][:, T_part:]
        #
        #         n_corrects_cp2 = np.mean(corrects_cp2, axis=1)
        #         n_mistakes_cp2 = np.mean(mistakes_cp2, axis=1)
        #         n_dks_cp2 = np.mean(dks_cp2, axis=1)
        #         sum_lure_act_cp2 = np.mean(
        #             np.mean(lure_act_cp2, axis=-1), axis=1)
        #
        #         dvs = [n_corrects_cp2, n_mistakes_cp2, n_dks_cp2]
        #         ylabels = ['% correct', '% mistake', '% uncertain']
        #         colors = [sns.color_palette()[0], sns.color_palette()
        #                   [3], 'grey']
        #
        #         f, axes = plt.subplots(
        #             1, len(dvs), figsize=(5 * len(dvs), 4.5))
        #         for i, ax in enumerate(axes):
        #             r_val, p_val = pearsonr(sum_lure_act_cp2, dvs[i])
        #             sns.regplot(
        #                 sum_lure_act_cp2, dvs[i], color=colors[i],
        #                 ax=axes[i]
        #             )
        #             ax.set_xlabel('Lure activation')
        #             ax.set_ylabel(ylabels[i])
        #             tt_ = r'$r \approx %.2f, p \approx %.3f$' % (r_val, p_val)
        #             ax.set_title(tt_)
        #         sns.despine()
        #         f.tight_layout()
        #         fig_path = os.path.join(fig_dir, f'reg-lure-behav-{cond_}.png')
        #         f.savefig(fig_path, dpi=150, bbox_to_anchor='tight')
        #
        #     '''use lure recall to predict what error the model make'''
        #
        #     # # np.argmax(np.mean(lure_act_dmp2, axis=1), axis=-1)
        #     # np.shape(Y)
        #     #
        #     # # get the memory ids for all memories
        #     # # index 0 - ongoing event; index 1 -> previous event; etc.
        #     # mem_ids = [np.where(cond_ids[cond_])[0] - i
        #     #            for i in range(n_event_remember)]
        #     #
        #     # # [i for i in range(n_event_remember)]
        #     # mem_ids[0]
        #     # # np.shape(mem_ids)
        #     # len(X_raw)
        #     # X_raw[0]
        #     # # extract observed features from prev events
        #     # i = 1
        #     # X_i = np.array(X_raw)[mem_ids[i]]
        #     #
        #     # X_i_ok = X_i[:, :, :task.k_dim]
        #     # X_i_ov = X_i[:, :, task.k_dim:task.k_dim+task.v_dim]
        #     # event_order = np.argmax(X_i_ok, axis=2)
        #     #
        #     # n_fv_observed = np.zeros((len(X_i), T_part))
        #     # # loop over events
        #     # for j in range(len(X_i)):
        #     #     event_order_j_p2 = event_order[j][T_part:]
        #     #     for t in range(T_part):
        #     #         n_fv_observed[j, t] = np.sum(event_order_j_p2[:t] <= t)
        #     #
        #     # np.shape(X_i_ok)
        #     # np.shape(np.argmax(X_i_ok, axis=2))
        #     #
        #     # i = 0
        #     #
        #     # np.array(X_raw)[mem_ids[i]]
        #
        #     '''compute q source, and check q source % for all conditions'''
        #
        #     # pick a condition
        #     q_source = get_qsource(true_dk_em, true_dk_wm, cond_ids, p)
        #     all_q_source = list(q_source['DM'].keys())
        #     q_source_list[i_s] = q_source
        #
        #     # split lca parameters by query source
        #     lca_param_dicts_bq = dict(
        #         zip(lca_param_names, [inpt_dict_bq, leak_dict_bq, comp_dict_bq]))
        #     for i_p, p_name in enumerate(lca_param_names):
        #         p_record_cond = lca_param_records[i_p][cond_ids['DM']]
        #         p_record_cond_qs = sep_by_qsource(
        #             p_record_cond, q_source['DM'], n_se=n_se)
        #         for qs in DM_qsources:
        #             lca_param_dicts_bq[p_name][qs][i_s] = p_record_cond_qs[qs][0]
        #
        #     # split target actvation by query source
        #     tma_qs = sep_by_qsource(
        #         avg_ma['DM']['targ'], q_source['DM'], n_se=n_se)
        #
        #     # plot distribution of  query source
        #     width = .85
        #     f, axes = plt.subplots(3, 1, figsize=(7, 10))
        #     for i, (cd_name, q_source_cd_p2) in enumerate(q_source.items()):
        #         # unpack data
        #         eo_cd_p2, wo_cd_p2, nt_cd_p2, bt_cd_p2 = q_source_cd_p2.values()
        #         axes[i].bar(range(n_param), prop_true(
        #             eo_cd_p2), label='EM', width=width)
        #         axes[i].bar(range(n_param), prop_true(wo_cd_p2), label='WM', width=width,
        #                     bottom=prop_true(eo_cd_p2))
        #         axes[i].bar(range(n_param), prop_true(bt_cd_p2), label='both', width=width,
        #                     bottom=prop_true(eo_cd_p2) + prop_true(wo_cd_p2))
        #         axes[i].bar(range(n_param), prop_true(nt_cd_p2), label='neither', width=width,
        #                     bottom=prop_true(eo_cd_p2) + prop_true(wo_cd_p2) + prop_true(bt_cd_p2))
        #         axes[i].set_ylabel('Proportion (%)')
        #         axes[i].set_title(f'{cd_name}')
        #         axes[i].legend()
        #         axes[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #     axes[-1].set_xlabel('Time (part 2)')
        #     sns.despine()
        #     f.tight_layout()
        #     fig_path = os.path.join(fig_dir, f'tz-q-source.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''plot target memory activation profile, for all trials'''
        #     cond_name = 'DM'
        #     m_type = 'targ'
        #     ylab = 'Activation'
        #
        #     f, axes = plt.subplots(2, 2, figsize=(9, 7))
        #     mu_, er_ = compute_stats(
        #         avg_ma[cond_name][m_type][:, T_part:], n_se=3
        #     )
        #     axes[0, 0].plot(
        #         avg_ma[cond_name][m_type][:, T_part:].T,
        #         alpha=.1, color=gr_pal[0]
        #     )
        #     axes[0, 0].errorbar(x=range(T_part), y=mu_,
        #                         yerr=er_, color='black')
        #     axes[0, 0].set_xlabel('Time (part 2)')
        #     axes[0, 0].set_ylabel(ylab)
        #     axes[0, 0].set_title(f'All trials')
        #
        #     n_trials_ = 5
        #
        #     trials_ = np.random.choice(
        #         range(len(avg_ma[cond_name][m_type])), n_trials_)
        #     axes[0, 1].plot(avg_ma[cond_name][m_type]
        #                     [:, T_part:][trials_, :].T)
        #
        #     axes[0, 1].set_xlabel('Time (part 2)')
        #     axes[0, 1].set_ylabel(ylab)
        #     axes[0, 1].set_title(f'{n_trials_} example trials')
        #     axes[0, 1].set_ylim(axes[0, 0].get_ylim())
        #
        #     sorted_targ_act_cond_p2 = np.sort(
        #         avg_ma[cond_name][m_type][:, T_part:], axis=1)[:, :: -1]
        #     mu_, er_ = compute_stats(sorted_targ_act_cond_p2, n_se=3)
        #     axes[1, 0].plot(sorted_targ_act_cond_p2.T,
        #                     alpha=.1, color=gr_pal[0])
        #     axes[1, 0].errorbar(x=range(T_part), y=mu_,
        #                         yerr=er_, color='black')
        #     axes[1, 0].set_ylabel(ylab)
        #     axes[1, 0].set_xlabel(f'Time (the sorting axis)')
        #     axes[1, 0].set_title(f'Sorted')
        #
        #     recall_peak_times = np.argmax(
        #         avg_ma[cond_name][m_type][:, T_part:], axis=1)
        #     sns.violinplot(recall_peak_times, color=gr_pal[0], ax=axes[1, 1])
        #     axes[1, 1].set_xlim(axes[0, 1].get_xlim())
        #     axes[1, 1].set_title(f'Max distribution')
        #     axes[1, 1].set_xlabel('Time (part 2)')
        #     axes[1, 1].set_ylabel('Density')
        #
        #     if pad_len_test > 0:
        #         axes[0, 0].axvline(pad_len_test, color='grey', linestyle='--')
        #         axes[0, 1].axvline(pad_len_test, color='grey', linestyle='--')
        #         axes[1, 1].axvline(pad_len_test, color='grey', linestyle='--')
        #     m_type_txt = 'target' if m_type == 'targ' else m_type
        #     f.suptitle(f'Memory activation profile, {m_type_txt}, {cond_name}',
        #                y=.95, fontsize=18)
        #
        #     sns.despine()
        #     f.tight_layout(rect=[0, 0, 1, 0.9])
        #     # f.subplots_adjust(top=0.9)
        #     fig_path = os.path.join(
        #         fig_dir, f'mem-act-profile-{cond_name}-{m_type}-lca.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''memory activation ~ time + dk'''
        #     cond_name = 'DM'
        #     m_type = 'targ'
        #     ylab = 'Activation'
        #
        #     # f, axes = plt.subplots(2, 2, figsize=(9, 7))
        #     np.shape(avg_ma[cond_name][m_type][:, T_part:])
        #     np.shape(q_source['DM']['EM only'])
        #
        #     '''recall - power law '''
        #     # mu_, er_ = compute_stats(
        #     #     avg_ma[cond_name][m_type][:, T_part:], n_se=3
        #     # )
        #     #
        #     # nthres = 50
        #     # t = 2
        #     # tma_min = np.min(avg_ma[cond_name][m_type][:, T_part:])
        #     # tma_max = np.max(avg_ma[cond_name][m_type][:, T_part:])
        #     # thres = np.linspace(tma_min, tma_max, nthres)
        #     # prob = np.zeros((T_part, nthres))
        #     #
        #     # for t in range(T_part):
        #     #     for i, thres_i in enumerate(thres):
        #     #         prob[t, i] = np.mean(
        #     #             avg_ma[cond_name][m_type][:, T_part+t] > thres_i)
        #     #
        #     # v_pal = sns.color_palette('Blues', n_colors=T_part)
        #     # sns.palplot(v_pal)
        #     #
        #     # f, ax = plt.subplots(1, 1, figsize=(5, 4))
        #     # for t in range(T_part):
        #     #     ax.plot(np.log(thres), prob[t, :], color=v_pal[t])
        #     # ax.set_ylabel('P(activation > v | t)')
        #     # ax.set_xlabel('log(v)')
        #     # f.tight_layout()
        #     # sns.despine()
        #
        #     '''use previous uncertainty to predict memory activation'''
        #     cond_name = 'DM'
        #     m_type = 'targ'
        #     dk_cond_p2 = dks[cond_ids[cond_name], n_param:]
        #     t_pick_max = 9
        #     t_picks = np.arange(2, t_pick_max)
        #     v_pal = sns.color_palette('viridis', n_colors=t_pick_max)
        #     f, ax = plt.subplots(1, 1, figsize=(8, 4))
        #     for t_pick_ in t_picks:
        #         # compute number of don't knows produced so far
        #         ndks_p2_b4recall = np.sum(dk_cond_p2[:, :t_pick_], axis=1)
        #         ndks_p2_b4recall = ndks_p2_b4recall / t_pick_
        #         nvs = np.unique(ndks_p2_b4recall)
        #         ma_mu = np.zeros(len(nvs),)
        #         ma_er = np.zeros(len(nvs),)
        #         for i, val in enumerate(np.unique(ndks_p2_b4recall)):
        #             ma_ndk = avg_ma[cond_name][m_type][:,
        #                                                T_part:][ndks_p2_b4recall == val, t_pick_]
        #             ma_mu[i], ma_er[i] = compute_stats(
        #                 ma_ndk, n_se=1)
        #         ax.errorbar(x=nvs, y=ma_mu, yerr=ma_er, color=v_pal[t_pick_])
        #
        #     ax.legend(t_picks, title='time', bbox_to_anchor=(1.3, 1.1))
        #     ax.set_title(f'Target activation, {cond_name}')
        #     # ax.set_xlabel('# don\'t knows')
        #     ax.set_xlabel('percent uncertain')
        #     ax.set_xlim([0, 1.05])
        #     ax.set_ylabel('average recall peak')
        #     sns.despine()
        #     f.tight_layout()
        #     fig_path = os.path.join(
        #         fig_dir, f'tz-{cond_name}-targact-by-propdk.png')
        #     # fig_path = os.path.join(fig_dir, f'tz-{cond_name}-targact-by-ndk.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''use CURRENT uncertainty to predict memory activation'''
        #     cond_name = 'DM'
        #     targ_act_cond_p2_stats = sep_by_qsource(
        #         avg_ma[cond_name]['targ'][:, T_part + pad_len_test:],
        #         q_source[cond_name],
        #         n_se=n_se
        #     )
        #
        #     for qs in DM_qsources:
        #         tma_dm_p2_dict_bq[qs][i_s] = targ_act_cond_p2_stats[qs][0]
        #
        #     f, ax = plt.subplots(1, 1, figsize=(7, 4))
        #     for key, [mu_, er_] in targ_act_cond_p2_stats.items():
        #         if not np.all(np.isnan(mu_)):
        #             ax.errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
        #     ax.set_ylabel(f'{param_name}')
        #     # ax.legend(fancybox=True)
        #     ax.set_title(f'Target memory activation, {cond_name}')
        #     ax.set_xlabel('Time (part 2)')
        #     ax.set_ylabel('Activation')
        #     # ax.set_ylim([-.05, .75])
        #     ax.set_xticks([0, p.env.n_param - 1])
        #     ax.legend(['not in WM', 'in WM'], fancybox=True)
        #     # ax.legend([])
        #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #     f.tight_layout()
        #     sns.despine()
        #     fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-qsource.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''
        #     predictions performance / dk / errors
        #     w.r.t prediction source (EM, WM)
        #     '''
        #     cond_name = 'DM'
        #     corrects_cond_p2 = corrects[cond_ids[cond_name], n_param:]
        #     mistakes_cond_p2 = mistakes[cond_ids[cond_name], n_param:]
        #     acc_cond_p2_stats = sep_by_qsource(
        #         corrects_cond_p2, q_source[cond_name], n_se=n_se)
        #     dk_cond_p2_stats = sep_by_qsource(
        #         dk_cond_p2, q_source[cond_name], n_se=n_se)
        #     mistakes_cond_p2_stats = sep_by_qsource(
        #         mistakes_cond_p2, q_source[cond_name], n_se=n_se)
        #
        #     stats_to_plot = {
        #         'correct': acc_cond_p2_stats, 'uncertain': dk_cond_p2_stats,
        #         'error': mistakes_cond_p2_stats,
        #     }
        #
        #     f, axes = plt.subplots(len(stats_to_plot), 1, figsize=(7, 10))
        #     # loop over all statistics
        #     for i, (stats_name, stat) in enumerate(stats_to_plot.items()):
        #         # loop over all q source
        #         for key, [mu_, er_] in stat.items():
        #             # plot if sample > 0
        #             if not np.all(np.isnan(mu_)):
        #                 axes[i].errorbar(
        #                     x=range(n_param), y=mu_, yerr=er_, label=key
        #                 )
        #         # for every panel/stats
        #         axes[i].set_ylabel(f'P({stats_name})')
        #         axes[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #         axes[i].set_ylim([-.05, 1.05])
        #     # for the entire panel
        #     axes[0].set_title(f'Performance, {cond_name}')
        #     axes[-1].legend(fancybox=True)
        #     axes[-1].set_xlabel('Time (part 2)')
        #     f.tight_layout()
        #     sns.despine()
        #     fig_path = os.path.join(
        #         fig_dir, f'tz-{cond_name}-stats-by-qsource.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''ma ~ correct in the EM only case'''
        #     cond_name = 'DM'
        #     m_type = 'targ'
        #
        #     tma_crt_mu, tma_crt_er = np.zeros(n_param,), np.zeros(n_param,)
        #     tma_incrt_mu, tma_incrt_er = np.zeros(n_param,), np.zeros(n_param,)
        #     for t in range(n_param):
        #         sel_op = q_source[cond_name]['EM only'][:, t]
        #         tma_ = avg_ma[cond_name][m_type][sel_op,
        #                                          T_part + t + pad_len_test]
        #         crt_ = corrects_cond_p2[q_source[cond_name]
        #                                 ['EM only'][:, t], t]
        #         tma_crt_mu[t], tma_crt_er[t] = compute_stats(tma_[crt_])
        #         tma_incrt_mu[t], tma_incrt_er[t] = compute_stats(tma_[~crt_])
        #
        #     f, ax = plt.subplots(1, 1, figsize=(7, 4))
        #     ax.errorbar(x=range(n_param), y=tma_crt_mu,
        #                 yerr=tma_crt_er, label='correct')
        #     ax.errorbar(x=range(n_param), y=tma_incrt_mu,
        #                 yerr=tma_incrt_er, label='incorrect')
        #     # ax.set_ylim([-.05, None])
        #     ax.legend()
        #     ax.set_title(f'Target memory activation, {cond_name}')
        #     ax.set_ylabel('Activation')
        #     ax.set_xlabel('Time (part 2)')
        #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #     sns.despine()
        #     f.tight_layout()
        #     fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-cic.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''ma ~ kdk in the EM only case'''
        #
        #     tma_k_mu, tma_k_er = np.zeros(n_param,), np.zeros(n_param,)
        #     tma_dk_mu, tma_dk_er = np.zeros(n_param,), np.zeros(n_param,)
        #     for t in range(n_param):
        #         sel_op = q_source[cond_name]['EM only'][:, t]
        #         tma_ = avg_ma[cond_name][m_type][sel_op,
        #                                          T_part + t + pad_len_test]
        #         dk_ = dk_cond_p2[q_source[cond_name]['EM only'][:, t], t]
        #         tma_k_mu[t], tma_k_er[t] = compute_stats(tma_[~dk_])
        #         tma_dk_mu[t], tma_dk_er[t] = compute_stats(tma_[dk_])
        #
        #     f, ax = plt.subplots(1, 1, figsize=(7, 4))
        #     ax.errorbar(x=range(n_param), y=tma_k_mu,
        #                 yerr=tma_k_er, label='know')
        #     ax.errorbar(x=range(n_param), y=tma_dk_mu,
        #                 yerr=tma_dk_er, label='don\'t know')
        #     ax.set_ylim([-.05, None])
        #     ax.legend()
        #     ax.set_title(f'Target memory activation, {cond_name}')
        #     ax.set_ylabel('Activation')
        #     ax.set_xlabel('Time (part 2)')
        #     sns.despine()
        #     f.tight_layout()
        #     fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-kdk.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''analyze the EM-only condition'''
        #
        #     for source_ in all_q_source:
        #         if np.all(np.isnan(acc_cond_p2_stats[source_][0])):
        #             continue
        #         f, ax = plt.subplots(1, 1, figsize=(6, 3.5))
        #         plot_pred_acc_rcl(
        #             acc_cond_p2_stats[source_][0], acc_cond_p2_stats[source_][1],
        #             acc_cond_p2_stats[source_][0] +
        #             dk_cond_p2_stats[source_][0],
        #             p, f, ax,
        #             title=f'Prediction performance, {source_}, {cond_name}',
        #             add_legend=True,
        #         )
        #         # if slience_recall_time is not None:
        #         #     ax.axvline(slience_recall_time, color='red',
        #         #                linestyle='--', alpha=alpha)
        #         ax.set_xlabel('Time (part 2)')
        #         ax.set_ylabel('Accuracy')
        #         ax.set_ylim([0, 1.05])
        #         f.tight_layout()
        #         sns.despine()
        #         fig_path = os.path.join(
        #             fig_dir, f'tz-pa-{cond_name}-{source_}.png')
        #         f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''compare the over time'''
        #
        #     def get_max_score(mem_act_cond):
        #         n_trials_ = np.shape(mem_act_cond)[0]
        #         rt_ = np.argmax(
        #             np.max(mem_act_cond[:, T_part:], axis=-1),
        #             axis=-1
        #         ) + T_part
        #         ms_targ = np.array(
        #             [np.max(mem_act_cond[i, rt_[i], :])
        #              for i in range(n_trials_)]
        #         )
        #         return ms_targ
        #
        #     ms_lure = get_max_score(sim_lca_dict['NM']['lure'])
        #     ms_targ = get_max_score(sim_lca_dict['DM']['targ'])
        #
        #     [dist_l, dist_r], [hist_info_l, hist_info_r] = get_hist_info(
        #         ms_lure, ms_targ)
        #     tpr, fpr = compute_roc(dist_l, dist_r)
        #     auc = metrics.auc(fpr, tpr)
        #
        #     # collect group data
        #     ms_lure_list[i_s] = ms_lure
        #     ms_targ_list[i_s] = ms_targ
        #     tpr_list[i_s] = tpr
        #     fpr_list[i_s] = fpr
        #     auc_list[i_s] = auc
        #
        #     [dist_l_edges, dist_l_normed, dist_l_edges_mids, bin_width_l] = hist_info_l
        #     [dist_r_edges, dist_r_normed, dist_r_edges_mids, bin_width_r] = hist_info_r
        #
        #     leg_ = ['NM', 'DM']
        #     f, axes = plt.subplots(
        #         1, 2, figsize=(10, 3.3), gridspec_kw={'width_ratios': [2, 1]}
        #     )
        #     axes[0].bar(dist_l_edges_mids, dist_l_normed, width=bin_width_l,
        #                 alpha=.5, color=gr_pal[1])
        #     axes[0].bar(dist_r_edges_mids, dist_r_normed, width=bin_width_r,
        #                 alpha=.5, color=gr_pal[0])
        #     axes[0].legend(leg_, frameon=True)
        #     axes[0].set_title('Max score distribution at recall')
        #     axes[0].set_xlabel('Recall strength')
        #     axes[0].set_ylabel('Probability')
        #     axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #     axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #
        #     axes[1].plot(fpr, tpr)
        #     axes[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
        #     axes[1].set_title('ROC, AUC = %.2f' % (auc))
        #     axes[1].set_xlabel('FPR')
        #     axes[1].set_ylabel('TPR')
        #     f.tight_layout()
        #     sns.despine()
        #     fig_path = os.path.join(fig_dir, f'ms-dist-t-peak.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''compute inter-event similarity'''
        #     targ_raw = np.argmax(np.array(Y_raw), axis=-1)
        #
        #     ambiguity = np.zeros((n_examples, n_event_remember - 1))
        #     for i in range(n_examples):
        #         cur_mem_ids = np.arange(i - n_event_remember + 1, i)
        #         for j, j_raw in enumerate(cur_mem_ids):
        #             ambiguity[i, j] = compute_event_similarity(
        #                 targ_raw[i], targ_raw[j])
        #
        #     # plot event similarity distribution
        #     confusion_mu = np.mean(ambiguity, axis=1)
        #
        #     f, axes = plt.subplots(2, 1, figsize=(5, 6))
        #     sns.distplot(confusion_mu, kde=False, ax=axes[0])
        #     axes[0].set_ylabel('P')
        #     axes[0].set_xlim([0, 1])
        #
        #     sns.distplot(np.ravel(ambiguity), kde=False, ax=axes[1])
        #     axes[1].set_xlabel('Parameter overlap')
        #     axes[1].set_ylabel('P')
        #     axes[1].set_xlim([0, 1])
        #
        #     sns.despine()
        #     f.tight_layout()
        #
        #     '''performance metrics ~ ambiguity'''
        #     corrects_by_cond, mistakes_by_cond, dks_by_cond = {}, {}, {}
        #     corrects_by_cond_mu, mistakes_by_cond_mu, dks_by_cond_mu = {}, {}, {}
        #     confusion_by_cond_mu = {}
        #     for cond_name, cond_ids_ in cond_ids.items():
        #         # print(cond_name, cond_ids_)
        #         # collect the regressor by condiiton
        #         confusion_by_cond_mu[cond_name] = confusion_mu[cond_ids_]
        #         # collect the performance metrics
        #         corrects_by_cond[cond_name] = corrects[cond_ids_, :]
        #         mistakes_by_cond[cond_name] = mistakes[cond_ids_, :]
        #         dks_by_cond[cond_name] = dks[cond_ids_, :]
        #         # compute average for the recall phase
        #         corrects_by_cond_mu[cond_name] = np.mean(
        #             corrects_by_cond[cond_name][:, T_part:], axis=1)
        #         mistakes_by_cond_mu[cond_name] = np.mean(
        #             mistakes_by_cond[cond_name][:, T_part:], axis=1)
        #         dks_by_cond_mu[cond_name] = np.mean(
        #             dks_by_cond[cond_name][:, T_part:], axis=1)
        #
        #     '''show regression model w/ ambiguity as the predictor
        #     average the performance during the 2nd part (across time)
        #     '''
        #     # predictor: inter-event similarity
        #     ind_var = confusion_by_cond_mu
        #     dep_vars = {
        #         'Corrects': corrects_by_cond_mu, 'Errors': mistakes_by_cond_mu,
        #         'Uncertain': dks_by_cond_mu
        #     }
        #     c_pal = sns.color_palette(n_colors=3)
        #     f, axes = plt.subplots(3, 3, figsize=(
        #         9, 8), sharex=True, sharey=True)
        #     for col_id, cond_name in enumerate(cond_ids.keys()):
        #         for row_id, info_name in enumerate(dep_vars.keys()):
        #             sns.regplot(
        #                 ind_var[cond_name], dep_vars[info_name][cond_name],
        #                 # robust=True,
        #                 scatter_kws={'alpha': .5, 'marker': '.', 's': 15},
        #                 x_jitter=.025, y_jitter=.05,
        #                 color=c_pal[col_id],
        #                 ax=axes[row_id, col_id]
        #             )
        #             corr, pval = pearsonr(
        #                 ind_var[cond_name], dep_vars[info_name][cond_name]
        #             )
        #             str_ = 'r = %.2f, p = %.2f' % (corr, pval)
        #             str_ = str_ + '*' if pval < .05 else str_
        #             str_ = cond_name + '\n' + str_ if row_id == 0 else str_
        #             axes[row_id, col_id].set_title(str_)
        #             axes[row_id, 0].set_ylabel(info_name)
        #             axes[row_id, col_id].set_ylim([-.05, 1.05])
        #
        #         axes[-1, col_id].set_xlabel('Similarity')
        #     sns.despine()
        #     f.tight_layout()
        #     fig_path = os.path.join(fig_dir, f'ambiguity-by-cond.png')
        #     f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''t-RDM: raw similarity'''
        #     # data = C
        #     # trsm = {}
        #     # for cond_name in cond_ids.keys():
        #     #     if np.sum(cond_ids[cond_name]) == 0:
        #     #         continue
        #     #     else:
        #     #         data_cond_ = data[cond_ids[cond_name], :, :]
        #     #         trsm[cond_name] = compute_trsm(data_cond_)
        #     #
        #     # f, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
        #     # for i, cond_name in enumerate(TZ_COND_DICT.values()):
        #     #     sns.heatmap(
        #     #         trsm[cond_name], cmap='viridis', square=True,
        #     #         xticklabels=5, yticklabels=5,
        #     #         ax=axes[i]
        #     #     )
        #     #     axes[i].axvline(T_part, color='red', linestyle='--')
        #     #     axes[i].axhline(T_part, color='red', linestyle='--')
        #     #     axes[i].set_title(f'TR-TR correlation, {cond_name}')
        #     #     axes[i].set_ylabel('Time')
        #     # axes[-1].set_xlabel('Time')
        #     # f.tight_layout()
        #     # fig_path = os.path.join(fig_dir, f'trdm-by-cond.png')
        #     # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
        #
        #     '''pca the deicison activity'''
        #
        #     n_pcs = 8
        #     data = DA
        #     cond_name = 'DM'
        #
        #     # fit PCA
        #     pca = PCA(n_pcs)
        #
        #     data_cond = data[cond_ids[cond_name], :, :]
        #     data_cond = data_cond[:, ts_predict, :]
        #     targets_cond = targets[cond_ids[cond_name]]
        #     mistakes_cond = mistakes_by_cond[cond_name]
        #     dks_cond = dks[cond_ids[cond_name], :]
        #
        #     # Loop over timepoints
        #     pca_cum_var_exp = np.zeros((np.sum(ts_predict), n_pcs))
        #     for t in range(np.sum(ts_predict)):
        #         data_pca = pca.fit_transform(data_cond[:, t, :])
        #         pca_cum_var_exp[t] = np.cumsum(pca.explained_variance_ratio_)
        #
        #         f, ax = plt.subplots(1, 1, figsize=(7, 5))
        #         # plot the data
        #         for y_val in range(p.y_dim):
        #             y_sel_op = y_val == targets_cond
        #             sel_op_ = np.logical_and(
        #                 ~dks[cond_ids[cond_name], t], y_sel_op[:, t])
        #             ax.scatter(
        #                 data_pca[sel_op_, 0], data_pca[sel_op_, 1],
        #                 marker='o', alpha=alpha,
        #             )
        #         ax.scatter(
        #             data_pca[dks[cond_ids[cond_name], t], 0],
        #             data_pca[dks[cond_ids[cond_name], t], 1],
        #             marker='o', color='grey', alpha=alpha,
        #         )
        #         legend_list = [f'choice {k}' for k in range(
        #             task.y_dim)] + ['uncertain']
        #         if np.sum(mistakes_cond[:, t]) > 0:
        #             legend_list += ['error']
        #             ax.scatter(
        #                 data_pca[mistakes_cond[:, t],
        #                          0], data_pca[mistakes_cond[:, t], 1],
        #                 facecolors='none', edgecolors='red',
        #             )
        #         # add legend
        #         ax.legend(legend_list, fancybox=True, bbox_to_anchor=(1, .5),
        #                   loc='center left')
        #         # mark the plot
        #         ax.set_xlabel('PC 1')
        #         ax.set_ylabel('PC 2')
        #         # ax.set_title(f'Pre-decision activity, time = {t}')
        #         ax.set_title(f'Decision activity')
        #         sns.despine(offset=10)
        #         f.tight_layout()
        #
        #     '''compare norms'''
        #     dk_norms = np.linalg.norm(data_cond[dks_cond], axis=1)
        #     ndk_norms = np.linalg.norm(data_cond[~dks_cond], axis=1)
        #     dk_norm_mu, dk_norm_se = compute_stats(dk_norms)
        #     ndk_norm_mu, ndk_norm_se = compute_stats(ndk_norms)
        #
        #     f, ax = plt.subplots(1, 1, figsize=(5, 5))
        #     xticks = range(2)
        #     ax.bar(x=xticks, height=[dk_norm_mu, ndk_norm_mu],
        #            yerr=np.array([dk_norm_se, ndk_norm_se]) * 3)
        #     ax.set_xticks(xticks)
        #     ax.set_xticklabels(['uncertain', 'certain'])
        #     ax.set_ylabel('Activity norm')
        #     f.tight_layout()
        #     sns.despine()
        #
        #     '''plot cumulative variance explained curve'''
        #     t = -1
        #     pc_id = 1
        #     f, ax = plt.subplots(1, 1, figsize=(6, 4))
        #     ax.plot(pca_cum_var_exp[t])
        #     # ax.set_title('First %d PCs capture %d%% of variance' %
        #     #              (pc_id+1, pca_cum_var_exp[t, pc_id]*100))
        #     ax.axvline(pc_id, color='grey', linestyle='--')
        #     ax.axhline(pca_cum_var_exp[t, pc_id], color='grey', linestyle='--')
        #     ax.set_ylim([None, 1.05])
        #     ytickval_ = ax.get_yticks()
        #     ax.set_yticklabels(['{:,.0%}'.format(x) for x in ytickval_])
        #     ax.set_xticks(np.arange(n_pcs))
        #     ax.set_xticklabels(np.arange(n_pcs) + 1)
        #     ax.set_ylabel('cum. var. exp.')
        #     ax.set_xlabel('Number of components')
        #     sns.despine(offset=5)
        #     f.tight_layout()
        #
        # # '''end of loop over subject'''
        #
        # gdata_dict = {
        #     'lca_param_dicts': lca_param_dicts,
        #     'auc_list': auc_list,
        #     'acc_dict': acc_dict,
        #     'dk_dict': dk_dict,
        #     'mis_dict': mis_dict,
        #     'lca_ma_list': ma_list,
        #     'cosine_ma_list': ma_cos_list,
        #
        # }
        # fname = f'p{penalty_train}-{penalty_test}-data.pkl'
        # gdata_outdir = 'temp/'
        # pickle_save_dict(gdata_dict, os.path.join(gdata_outdir, fname))
        #
        # '''group level performance'''
        # n_se = 1
        # f, axes = plt.subplots(1, 3, figsize=(14, 4))
        # for i, cn in enumerate(all_conds):
        #     if i == 0:
        #         add_legend = True
        #         legend_loc = (.285, .7)
        #     else:
        #         add_legend = False
        #     # plot
        #     vs_ = [v_ for v_ in acc_dict[cn]['mu'] if v_ is not None]
        #     acc_gmu_, acc_ger_ = compute_stats(vs_, n_se=n_se, axis=0)
        #     vs_ = [v_ for v_ in dk_dict[cn]['mu'] if v_ is not None]
        #     dk_gmu_ = np.mean(vs_, axis=0)
        #     plot_pred_acc_rcl(
        #         acc_gmu_[T_part:], acc_ger_[T_part:],
        #         acc_gmu_[T_part:] + dk_gmu_[T_part:],
        #         p, f, axes[i],
        #         title=f'{cn}',
        #         add_legend=add_legend, legend_loc=legend_loc,
        #     )
        #     axes[i].set_ylim([0, 1.05])
        #     axes[i].set_xlabel('Time (part 2)')
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-acc.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # '''group level input gate by condition'''
        # n_se = 1
        # f, ax = plt.subplots(1, 1, figsize=(
        #     5 * (pad_len_test / n_param + 1), 4))
        # for i, cn in enumerate(all_conds):
        #     p_dict = lca_param_dicts[0]
        #     p_dict_ = remove_none(p_dict[cn]['mu'])
        #     mu_, er_ = compute_stats(p_dict_, n_se=n_se, axis=0)
        #     ax.errorbar(
        #         x=np.arange(T_part) - pad_len_test, y=mu_[T_part:], yerr=er_[T_part:], label=f'{cn}'
        #     )
        # ax.legend()
        # ax.set_ylim([-.05, .7])
        # # ax.set_ylim([-.05, .9])
        # ax.set_ylabel(lca_param_names[0])
        # ax.set_xlabel('Time (part 2)')
        # ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # if pad_len_test > 0:
        #     ax.axvline(0, color='grey', linestyle='--')
        # sns.despine()
        # f.tight_layout()
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-ig.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # n_se = 1
        # f, ax = plt.subplots(1, 1, figsize=(
        #     5 * (pad_len_test / n_param + 1), 4))
        # for i, cn in enumerate(['RM', 'DM']):
        #     p_dict = lca_param_dicts[0]
        #     p_dict_ = remove_none(p_dict[cn]['mu'])
        #     mu_, er_ = compute_stats(p_dict_, n_se=n_se, axis=0)
        #     ax.errorbar(
        #         x=np.arange(T_part) - pad_len_test, y=mu_[T_part:], yerr=er_[T_part:], label=f'{cn}'
        #     )
        #
        # ax.legend()
        # ax.set_ylim([-.05, .7])
        # # ax.set_ylim([-.05, .9])
        # ax.set_ylabel(lca_param_names[0])
        # ax.set_xlabel('Time (part 2)')
        # ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # if pad_len_test > 0:
        #     ax.axvline(0, color='grey', linestyle='--')
        # sns.despine()
        # f.tight_layout()
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-ig-nonm.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # '''group level LCA parameter by q source'''
        # # sns.set(style='whitegrid', palette='colorblind', context='poster')
        # n_se = 1
        # f, axes = plt.subplots(2, 1, figsize=(7, 6))
        # for i_p, p_name in enumerate(lca_param_names):
        #     for qs in ['EM only', 'both']:
        #         lca_param_dicts_bq_ = remove_none(
        #             lca_param_dicts_bq[p_name][qs]
        #         )
        #         mu_, er_ = compute_stats(
        #             lca_param_dicts_bq_, n_se=n_se, axis=0
        #         )
        #         axes[i_p].errorbar(
        #             x=range(T_part), y=mu_, yerr=er_, label=qs
        #         )
        # for i, ax in enumerate(axes):
        #     ax.legend()
        #     ax.set_ylabel(lca_param_names[i])
        #     ax.set_xlabel('Time (part 2)')
        #     ax.set_xticks(np.arange(0, p.env.n_param, 5))
        #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # sns.despine()
        # f.tight_layout()
        #
        # '''group level memory activation by condition'''
        # # sns.set(style='white', palette='colorblind', context='talk')
        # n_se = 1
        # ma_list_dict = {'lca': ma_list, 'cosine': ma_cos_list}
        # for metric_name, ma_list_ in ma_list_dict.items():
        #     # ma_list_ = ma_list
        #     # ma_list_ = ma_cos_list
        #     # f, axes = plt.subplots(3, 1, figsize=(7, 9))
        #     f, axes = plt.subplots(1, 3, figsize=(14, 4))
        #     for i, c_name in enumerate(cond_ids.keys()):
        #         for m_type in memory_types:
        #             if m_type == 'targ' and c_name == 'NM':
        #                 continue
        #             color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
        #
        #             # for the current cn - mt combination, average across people
        #             y_list = []
        #             for i_s, subj_id in enumerate(subj_ids):
        #                 if ma_list_[i_s] is not None:
        #                     ma_list_i_s = ma_list_[i_s]
        #                     y_list.append(
        #                         ma_list_i_s[c_name][m_type]['mu'][T_part:]
        #                     )
        #             mu_, er_ = compute_stats(y_list, n_se=1, axis=0)
        #             axes[i].errorbar(
        #                 x=range(T_part), y=mu_, yerr=er_,
        #                 label=f'{m_type}', color=color_
        #             )
        #         axes[i].set_title(c_name)
        #         axes[i].set_xlabel('Time (part 2)')
        #     axes[0].set_ylabel('Memory activation')
        #     # make all ylims the same
        #     ylim_l, ylim_r = get_ylim_bonds(axes)
        #     for i, ax in enumerate(axes):
        #         ax.legend()
        #         ax.set_xlabel('Time (part 2)')
        #         ax.set_ylim([np.max([-.05, ylim_l]), ylim_r])
        #         ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        #         ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #         ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #         if metric_name == 'lca':
        #             ax.set_yticks([0, .2, .4])
        #             ax.set_ylim([-.01, .45])
        #             # ax.set_yticks([0, .2, .4, .6])
        #             # ax.set_ylim([-.01, .75])
        #         else:
        #             ax.set_yticks([0, .5, 1])
        #             ax.set_ylim([-.05, 1.05])
        #
        #     if pad_len_test > 0:
        #         for ax in axes:
        #             ax.axvline(pad_len_test, color='grey', linestyle='--')
        #     f.tight_layout()
        #     sns.despine()
        #
        #     fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-{metric_name}-rs.png'
        #     f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # '''ma raw'''
        # # ma_raw_list = remove_none(ma_raw_list)
        # #
        # # # cond_ = 'DM'
        # # f, axes = plt.subplots(1, 3, figsize=(14, 4))
        # # for i, cond_ in enumerate(cond_ids.keys()):
        # #     if cond_ is not 'NM':
        # #         targ_acts = np.array([ma_raw_i[cond_]['targ']
        # #                               for ma_raw_i in ma_raw_list])
        # #         np.shape(np.squeeze(targ_acts[0]))
        # #         targ_acts_mu = np.mean(np.mean(targ_acts, axis=0), axis=0)
        # #         targ_acts_se = np.std(np.mean(targ_acts, axis=1),
        # #                               axis=0) / np.sqrt(len(ma_raw_list))
        # #
        # #     lure_acts = np.array([ma_raw_i[cond_]['lure']
        # #                           for ma_raw_i in ma_raw_list])
        # #     lure_acts_mu = np.mean(np.mean(lure_acts, axis=0), axis=0)
        # #     lure_acts_se = np.std(np.mean(lure_acts, axis=1),
        # #                           axis=0) / np.sqrt(len(ma_raw_list))
        # #
        # #     alphas = [.5, 1]
        # #
        # #     for mid in range(2):
        # #         if cond_ is not 'NM':
        # #             axes[i].errorbar(
        # #                 y=targ_acts_mu[T_part:, mid],
        # #                 yerr=targ_acts_se[T_part:, mid],
        # #                 x=range(T_part), color=gr_pal[0], alpha=alphas[mid]
        # #             )
        # #         axes[i].errorbar(
        # #             y=lure_acts_mu[T_part:, mid],
        # #             yerr=lure_acts_se[T_part:, mid],
        # #             x=range(T_part), color=gr_pal[1], alpha=alphas[mid]
        # #         )
        # #     axes[i].set_ylim([-.05, .6])
        # #     axes[i].set_xlabel('Time (part 2)')
        # # axes[0].set_ylabel('Memory activation')
        # # f.tight_layout()
        # # sns.despine()
        #
        # '''target memory activation by q source'''
        # n_se = 1
        # f, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        # for qs in DM_qsources:
        #     # remove none
        #     tma_dm_p2_dict_bq_ = remove_none(tma_dm_p2_dict_bq[qs])
        #     mu_, er_ = compute_stats(tma_dm_p2_dict_bq_, n_se=n_se, axis=0)
        #     ax.errorbar(
        #         x=range(T_part), y=mu_, yerr=er_, label=qs
        #     )
        # ax.set_ylabel('Memory activation')
        # ax.set_xlabel('Time (part 2)')
        # ax.legend(['not in WM', 'in WM'])
        # ax.set_xticks(np.arange(0, p.env.n_param, p.env.n_param - 1))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #
        # ax.set_yticks([0, .2, .4])
        # ax.set_ylim([-.01, .45])
        #
        # f.tight_layout()
        # sns.despine()
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-rs-dm-byq.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # # ms_lure = get_max_score(sim_lca_dict['NM']['lure'])
        # # ms_targ = get_max_score(sim_lca_dict['DM']['targ'])
        # # def remove_none(input_list):
        # #     return [i for i in input_list if i is not None]
        #
        # ms_lure_list = remove_none(ms_lure_list)
        # ms_targ_list = remove_none(ms_targ_list)
        # tpr_list = remove_none(tpr_list)
        # fpr_list = remove_none(fpr_list)
        # auc_list = remove_none(auc_list)
        # auc_1se = np.std(auc_list) / np.sqrt(len(auc_list))
        # # ms_targ_list = [i for i in ms_targ_list if i != None]
        # [dist_l, dist_r], [hist_info_l, hist_info_r] = get_hist_info(
        #     np.concatenate(ms_lure_list),
        #     np.concatenate(ms_targ_list)
        # )
        # tpr_g, fpr_g = compute_roc(dist_l, dist_r)
        # auc_g = metrics.auc(tpr_g, fpr_g)
        #
        # [dist_l_edges, dist_l_normed, dist_l_edges_mids, bin_width_l] = hist_info_l
        # [dist_r_edges, dist_r_normed, dist_r_edges_mids, bin_width_r] = hist_info_r
        #
        # leg_ = ['NM', 'DM']
        # f, axes = plt.subplots(
        #     1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]}
        # )
        # axes[0].bar(dist_l_edges_mids, dist_l_normed, width=bin_width_l,
        #             alpha=.5, color=gr_pal[1])
        # axes[0].bar(dist_r_edges_mids, dist_r_normed, width=bin_width_r,
        #             alpha=.5, color=gr_pal[0])
        # axes[0].legend(leg_, frameon=True)
        # # axes[0].set_title('Max score distribution at recall')
        # axes[0].set_xlabel('Max score')
        # axes[0].set_ylabel('Probability')
        # axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #
        # axes[1].plot(fpr_g, tpr_g)
        # # for tpr_list_i, fpr_list_i in zip(tpr_list, fpr_list):
        # #     plt.plot(fpr_list_i, tpr_list_i, color=c_pal[0], alpha=.15)
        # axes[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
        # axes[1].set_title('AUC = %.2f' % (np.mean(auc_list)))
        # axes[1].set_xlabel('FPR')
        # axes[1].set_ylabel('TPR')
        # axes[1].set_xticks([0, 1])
        # axes[1].set_yticks([0, 1])
        # f.tight_layout()
        # sns.despine()
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-roc.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # '''auc ~ center of mass of input gate'''
        # n_se = 1
        # cn = 'DM'
        # p_dict_ = remove_none(lca_param_dicts[0][cn]['mu'])
        # ig_p2 = np.array(p_dict_)[:, T_part:]
        # ig_p2_norm = ig_p2
        # rt = np.dot(ig_p2_norm, (np.arange(T_part) + 1))
        # r_val, p_val = pearsonr(rt, np.array(auc_list))
        #
        # f, ax = plt.subplots(1, 1, figsize=(5, 4))
        # sns.regplot(rt, auc_list)
        # ax.set_xlabel('Recall time')
        # ax.set_ylabel('AUC')
        # ax.annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val), xy=(
        #     0.05, 0.05), xycoords='axes fraction')
        # sns.despine()
        # f.tight_layout()
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-reg-rt-auc.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # def compute_corr2behav(behav_data):
        #     cond = 'DM'
        #     temp = deepcopy(behav_data[cond]['mu'])
        #     temp = remove_none(temp)
        #     bd_p2 = np.array(temp)[:, T_part:]
        #     bd_p2_mu = np.mean(bd_p2, axis=1)
        #     r_val, p_val = pearsonr(rt, bd_p2_mu)
        #     return bd_p2_mu, r_val, p_val
        #
        # behav_data_dict = {
        #     'Accuracy': acc_dict, 'Don\'t know': dk_dict, 'Mistakes': mis_dict,
        # }
        # cb_pal = sns.color_palette('colorblind', n_colors=4)
        # # sns.palplot(cb_pal)
        # behav_colors = [cb_pal[0], 'grey', cb_pal[3]]
        # f, axes = plt.subplots(3, 1, figsize=(5, 12))
        # for i, (behav_data_name, behav_data) in enumerate(
        #         behav_data_dict.items()):
        #
        #     bd_p2_mu, r_val, p_val = compute_corr2behav(behav_data)
        #     sns.regplot(rt, bd_p2_mu, ax=axes[i], color=behav_colors[i])
        #     axes[i].set_xlabel('Recall time')
        #     axes[i].set_ylabel(behav_data_name)
        #     # axes[i].axvline(0, linestyle='--', color='grey')
        #     axes[i].set_xlim([0, None])
        #     axes[i].annotate(
        #         r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val),
        #         xy=(0.05, 0.05), xycoords='axes fraction'
        #     )
        # sns.despine()
        # f.tight_layout()
        #
        # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-reg-rt-behav.png'
        # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # '''p(error | schema consistent) vs. p(error | schema inconsistent)'''
        #
        # if n_def_tps > 0:
        #     mis_dict_mu = {cn: None for cn in all_conds}
        #     missing_ids = {cn: None for cn in all_conds}
        #     for i, cn in enumerate(all_conds):
        #         # print(cn)
        #         mis_dict_mu[cn], missing_ids[cn] = remove_none(
        #             mis_dict[cn]['mu'], return_missing_idx=True
        #         )
        #         mis_dict_mu[cn] = np.array(mis_dict_mu[cn])[:, T_part:]
        #
        #     if len(missing_ids['DM']) > 0:
        #         # this is wrong, earlier poped ideas shift the indexing
        #         for missing_subject_id in missing_ids['DM']:
        #             def_tps_list.pop(missing_subject_id)
        #
        #     sc_sel_op = np.array(def_tps_list).astype(np.bool)
        #     f, axes = plt.subplots(1, 3, figsize=(17, 5))
        #     for i, cn in enumerate(all_conds):
        #         mis_gsc, mis_gsic = [], []
        #         for i_s in range(len(mis_dict_mu[cn])):
        #             mis_gsc.append(mis_dict_mu[cn][i_s][sc_sel_op[i_s]])
        #             mis_gsic.append(mis_dict_mu[cn][i_s][~sc_sel_op[i_s]])
        #         mis_gsc_mu, mis_gsc_se = compute_stats(
        #             np.mean(mis_gsc, axis=1))
        #         mis_gsic_mu, mis_gsic_se = compute_stats(
        #             np.mean(mis_gsic, axis=1))
        #
        #         heights = [mis_gsc_mu, mis_gsic_mu]
        #         yerrs = [mis_gsc_se, mis_gsic_se]
        #         xticklabels = ['yes', 'no']
        #         xticks = range(len(heights))
        #
        #         # f, ax = plt.subplots(1, 1, figsize=(6, 5))
        #         axes[i].bar(
        #             x=xticks, height=heights, yerr=yerrs,
        #             color=sns.color_palette('colorblind')[3]
        #         )
        #         axes[i].axhline(0, color='grey', linestyle='--')
        #         axes[i].set_title(cn)
        #         axes[i].set_xlabel('Has a prototypical event?')
        #         axes[i].set_xticks(xticks)
        #         axes[i].set_ylim([-.05, .5])
        #         axes[i].set_ylim([-.015, .15])
        #         axes[i].set_xticklabels(xticklabels)
        #         axes[i].set_ylabel('P(error)')
        #         f.tight_layout()
        #         sns.despine()
        #
        #     fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-error-schema-effect.png'
        #     f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # '''input gate | schema consistent vs. schema inconsistent'''
        #
        # def split_data_byschema(inp_data):
        #     d_gsc, d_gsic = [], []
        #     for i_s in range(len(inp_data)):
        #         d_gsc.append(inp_data[i_s][sc_sel_op[i_s]])
        #         d_gsic.append(inp_data[i_s][~sc_sel_op[i_s]])
        #     d_gsc_mu, d_gsc_se = compute_stats(np.mean(d_gsc, axis=1))
        #     d_gsic_mu, d_gsic_se = compute_stats(np.mean(d_gsic, axis=1))
        #     heights = [d_gsc_mu, d_gsic_mu]
        #     yerrs = [d_gsc_se, d_gsic_se]
        #     raw_data = [d_gsc, d_gsic]
        #     return heights, yerrs, raw_data
        #
        #     # for ii in range(n_subjs):
        #     #     sc_sel_op[ii] = np.roll(sc_sel_op[ii], -1, axis=None)
        # raw_data = {cn: None for cn in all_conds}
        # if n_def_tps > 0:
        #     n_se = 1
        #     # f, axes = plt.subplots(1, 2, figsize=(10, 4))
        #     f, axes = plt.subplots(1, 3, figsize=(17, 5))
        #     for i, cn in enumerate(all_conds):
        #         ig = lca_param_dicts[0][cn]['mu']
        #         ig = np.array(remove_none(ig))[:, T_part:]
        #         heights, yerrs, raw_data[cn] = split_data_byschema(ig)
        #         # f, ax = plt.subplots(1, 1, figsize=(6, 5))
        #         axes[i].bar(
        #             x=xticks, height=heights, yerr=yerrs,
        #             color=sns.color_palette('colorblind')[0]
        #         )
        #         axes[i].axhline(0, color='grey', linestyle='--')
        #         axes[i].set_title(cn)
        #         axes[i].set_xlabel('Has a prototypical event?')
        #         axes[i].set_xticks(xticks)
        #         axes[i].set_ylim([-.025, .2])
        #         axes[i].set_xticklabels(xticklabels)
        #         axes[i].set_ylabel('Input gate')
        #         f.tight_layout()
        #         sns.despine()
        #     fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-ig-schema-effect.png'
        #     f.savefig(fname, dpi=120, bbox_to_anchor='tight')
        #
        # # plt.plot(np.mean(ig, axis=1), np.mean(mis_dict_mu[cn], axis=1), 'x')
        # # cn = 'DM'
        # # mis_gsc, mis_gsic = [], []
        # # for i_s in range(len(mis_dict_mu[cn])):
        # #     mis_gsc.append(mis_dict_mu[cn][i_s][sc_sel_op[i_s]])
        # #     mis_gsic.append(mis_dict_mu[cn][i_s][~sc_sel_op[i_s]])
        # # mis_gsc_mu, mis_gsc_se = compute_stats(
        # #     np.mean(mis_gsc, axis=1))
        # # mis_gsic_mu, mis_gsic_se = compute_stats(
        # #     np.mean(mis_gsic, axis=1))
        # #
        # # d_gsc, d_gsic = raw_data[cn]
        # # f, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
        # # r_hp, p_hp = pearsonr(np.mean(d_gsc, axis=1), np.mean(mis_gsc, axis=1))
        # # r_np, p_np = pearsonr(np.mean(d_gsic, axis=1),
        # #                       np.mean(mis_gsic, axis=1))
        # # sns.regplot(
        # #     np.mean(d_gsc, axis=1), np.mean(mis_gsc, axis=1),
        # #     ax=axes[0]
        # # )
        # # sns.regplot(
        # #     np.mean(d_gsic, axis=1), np.mean(mis_gsic, axis=1),
        # #     ax=axes[1]
        # # )
        # # axes[0].set_xlabel('Input gate')
        # # axes[1].set_xlabel('Input gate')
        # # axes[0].set_ylabel('P(error)')
        # # # axes[1].set_ylabel('% Mistake')
        # # axes[0].set_title('Has a prototypical event')
        # # axes[1].set_title('No prototypical event')
        # #
        # # axes[0].annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_hp, p_hp),
        # #                  xy=(0.05, 0.05), xycoords='axes fraction')
        # # axes[1].annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_np, p_np),
        # #                  xy=(0.05, 0.05), xycoords='axes fraction')
        # # sns.despine()
        # # f.tight_layout()
        # #
        # # f, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        # # cn = 'DM'
        # # ig = lca_param_dicts[0][cn]['mu']
        # # ig = np.array(remove_none(ig))[:, T_part:]
        # # heights, yerrs, raw_data[cn] = split_data_byschema(ig)
        # #
        # # axes[0].bar(
        # #     x=xticks, height=heights, yerr=yerrs,
        # #     color=sns.color_palette('colorblind')[0]
        # # )
        # # axes[0].axhline(0, color='grey', linestyle='--')
        # # # axes[0].set_title(cn)
        # # # axes[0].set_xlabel('Has a prototypical event?')
        # # axes[0].set_xticks(xticks)
        # # axes[0].set_ylim([-.019, .19])
        # # axes[0].set_xticklabels(xticklabels)
        # # axes[0].set_ylabel('Input gate')
        # # f.tight_layout()
        # # sns.despine()
        # #
        # # mis_gsc, mis_gsic = [], []
        # # for i_s in range(len(mis_dict_mu[cn])):
        # #     mis_gsc.append(mis_dict_mu[cn][i_s][sc_sel_op[i_s]])
        # #     mis_gsic.append(mis_dict_mu[cn][i_s][~sc_sel_op[i_s]])
        # # mis_gsc_mu, mis_gsc_se = compute_stats(
        # #     np.mean(mis_gsc, axis=1))
        # # mis_gsic_mu, mis_gsic_se = compute_stats(
        # #     np.mean(mis_gsic, axis=1))
        # #
        # # heights = [mis_gsc_mu, mis_gsic_mu]
        # # yerrs = [mis_gsc_se, mis_gsic_se]
        # # xticklabels = ['yes', 'no']
        # # xticks = range(len(heights))
        # # axes[1].bar(
        # #     x=xticks, height=heights, yerr=yerrs,
        # #     color=sns.color_palette('colorblind')[3]
        # # )
        # # axes[1].axhline(0, color='grey', linestyle='--')
        # # # axes[1].set_title(cn)
        # # axes[1].set_xlabel('Has a prototypical event?')
        # # axes[1].set_xticks(xticks)
        # # axes[1].set_ylim([-.005, .05])
        # # axes[1].set_xticklabels(xticklabels)
        # # axes[1].set_ylabel('P(error)')
        # # f.tight_layout()
        # # sns.despine()
        # # fname = f'../figs/{exp_name}/p{penalty_train}-{penalty_test}-comb-schema-effect.png'
        # # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
