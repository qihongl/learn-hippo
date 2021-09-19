import dabest
import pandas as pd
import os
import numpy as np
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
# constants
lca_pnames = {0: 'input gate', 1: 'competition'}
all_conds = list(TZ_COND_DICT.values())
# n_param = 16
cpal = sns.color_palette()[:2]
f, ax = plt.subplots(1, 1, figsize=(6, 5.5))
exp_names = ['low-similarity', 'high-similarity']

for ei, exp_name in enumerate(exp_names):
    # exp_name = 'low-similarity'
    penalty_train = 4
    penaltys_test = [0, 2, 4]

    n_param = 16
    n_subjs = 15
    subj_ids = np.arange(n_subjs)
    penalty_random = 1
    def_prob = .25
    n_def_tps = 0
    comp_val = .8
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

    p = P(
        exp_name=exp_name, sup_epoch=supervised_epoch,
        n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
        enc_size=enc_size, n_event_remember=n_event_remember,
        def_prob=def_prob, n_def_tps=n_def_tps,
        penalty=penalty_train, penalty_random=penalty_random,
        p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
        cmpt=comp_val,
    )

    '''load data'''
    lca_param = {ptest: None for ptest in penaltys_test}
    auc = {ptest: None for ptest in penaltys_test}
    acc = {ptest: None for ptest in penaltys_test}
    mis = {ptest: None for ptest in penaltys_test}
    dk = {ptest: None for ptest in penaltys_test}
    ma_lca = defaultdict()
    ma_cosine = defaultdict()

    # for penalty_train in penaltys_train:

    for ptest in penaltys_test:
        print(f'penalty_train={penalty_train}, ptest={ptest}')
        # create logging dirs
        log_path, _ = build_log_path(
            0, p, log_root=log_root, mkdir=False, verbose=False
        )
        # load data
        dir_all_subjs = os.path.dirname(log_path)
        fname = 'p%d-%d.pkl' % (penalty_train, ptest)
        data = pickle_load_dict(os.path.join(dir_all_subjs, fname))
        # unpack data
        lca_param[ptest] = data['lca_param_dicts']
        auc[ptest] = data['auc_list']
        acc[ptest] = data['acc_dict']
        mis[ptest] = data['mis_dict']
        dk[ptest] = data['dk_dict']
        ma_lca[ptest] = data['lca_ma_list']
        ma_cosine[ptest] = data['cosine_ma_list']

    n_subjs_total = len(auc[ptest])
    # process the data - identify missing subjects
    missing_subjects = []
    for ptest in penaltys_test:
        for lca_pid, lca_pname in lca_pnames.items():
            for cond in all_conds:
                _, missing_ids_ = remove_none(
                    lca_param[ptest][lca_pid][cond]['mu'], return_missing_idx=True
                )
                missing_subjects.extend(missing_ids_)
    missing_subjects = np.unique(missing_subjects)

    n_subjs = n_subjs_total - len(missing_subjects)

    # process the data - remove missing subjects for all data dicts
    for i_ms in sorted(missing_subjects, reverse=True):
        # print(i_ms)
        for ptest in penaltys_test:
            del auc[ptest][i_ms]
            for cond in all_conds:
                del acc[ptest][cond]['mu'][i_ms]
                del mis[ptest][cond]['mu'][i_ms]
                del dk[ptest][cond]['mu'][i_ms]
                del acc[ptest][cond]['er'][i_ms]
                del mis[ptest][cond]['er'][i_ms]
                del dk[ptest][cond]['er'][i_ms]
                for lca_pid, lca_pname in lca_pnames.items():
                    del lca_param[ptest][lca_pid][cond]['mu'][i_ms]
                    del lca_param[ptest][lca_pid][cond]['er'][i_ms]
            del ma_lca[ptest][i_ms]

    '''compute average meory activation'''
    ma_dmp2, sum_ma_dmp2 = defaultdict(), defaultdict()
    ma_dmp2_mu, ma_dmp2_se = defaultdict(), defaultdict()

    for p_test in penaltys_test:
        ma_dmp2[p_test] = np.array(
            [ma_lca[p_test][s]['DM']['targ']['mu'][n_param:]
             for s in range(n_subjs)]
        )
        sum_ma_dmp2[p_test] = np.sum(ma_dmp2[p_test], axis=1)
        ma_dmp2_mu[p_test], ma_dmp2_se[p_test] = compute_stats(
            # ma_dmp2[p_test]
            np.mean(ma_dmp2[p_test], axis=1)
        )

    ax.errorbar(
        x=range(len(penaltys_test)),
        y=list(ma_dmp2_mu.values()), yerr=list(ma_dmp2_se.values()),
        label=exp_name, color=cpal[ei]
    )

    ax.set_xticks(range(len(penaltys_test)))
    ax.set_xticklabels(penaltys_test)
    ax.set_xlabel('Penalty, test')
    ax.set_ylabel('Average memory activation')
    ax.legend()
    sns.despine()
    f.tight_layout()
