import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, pickle_load_dict
from analysis import compute_acc, compute_dk, compute_stats, remove_none

warnings.filterwarnings("ignore")
sns.set(style='white', palette='colorblind', context='poster')
log_root = '../log/'

all_conds = TZ_COND_DICT.values()

# the name of the experiemnt
exp_name = 'tune-cmpt'
penalty_train = 4
penaltys_test = [0, 2, 4]
# penalty_test = 4
comp_vals = [0.0, .2, .4, .6, .8, 1.0]

n_subjs = 15
subj_ids = np.arange(n_subjs)
penalty_random = 1
def_prob = .25
n_def_tps = 0
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


f, ax = plt.subplots(1, 1, figsize=(8, 5))
blues = sns.color_palette('Blues', n_colors=len(penaltys_test))

for pi, penalty_test in enumerate(penaltys_test):

    def list_prealloc(n):
        return [{cond: np.zeros((len(comp_vals), n_param)) for cond in all_conds}
                for n in range(n)]
    auc_mu, auc_se = np.zeros(len(comp_vals),), np.zeros(len(comp_vals),)
    mat_mu, mat_se = np.zeros(len(comp_vals),), np.zeros(len(comp_vals),)
    mal_mu, mal_se = np.zeros(len(comp_vals),), np.zeros(len(comp_vals),)
    mad_mu, mad_se = np.zeros(len(comp_vals),), np.zeros(len(comp_vals),)
    rwd_mu = {cond: np.zeros((len(comp_vals), )) for cond in all_conds}
    rwd_se = {cond: np.zeros((len(comp_vals), )) for cond in all_conds}
    [acc_mu, acc_se, mis_mu, mis_se] = list_prealloc(4)

    for i, comp_val in enumerate(comp_vals):
        p = P(
            exp_name=exp_name, sup_epoch=supervised_epoch,
            n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
            enc_size=enc_size, n_event_remember=n_event_remember,
            def_prob=def_prob, n_def_tps=n_def_tps,
            penalty=penalty_train, penalty_random=penalty_random,
            p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
            cmpt=comp_val,
        )
        # create logging dirs
        log_path, _ = build_log_path(
            0, p, log_root=log_root, mkdir=False, verbose=False
        )
        # load data
        dir_all_subjs = os.path.dirname(log_path)
        fname = 'p%d-%d.pkl' % (penalty_train, penalty_test)
        data = pickle_load_dict(os.path.join(dir_all_subjs, fname))
        # get data
        acc_dict = data['acc_dict']
        dk_dict = data['dk_dict']
        mis_dict = data['mis_dict']
        ma_list = data['lca_ma_list']
        auc_list = data['auc_list']
        n_subjs_ = len(remove_none(data['acc_dict']['DM']['mu']))

        auc_mu[i], auc_se[i] = compute_stats(remove_none(auc_list))
        np.shape(remove_none(ma_list))

        for cond in all_conds:
            acc_p2 = np.array(remove_none(acc_dict[cond]['mu']))[:, n_param:]
            mis_p2 = np.array(remove_none(mis_dict[cond]['mu']))[:, n_param:]
            rwd_p2 = acc_p2 - mis_p2 * penalty_test
            acc_mu[cond][i], acc_se[cond][i] = compute_stats(acc_p2, axis=0)
            mis_mu[cond][i], mis_se[cond][i] = compute_stats(mis_p2, axis=0)
            rwd_mu[cond][i], rwd_se[cond][i] = compute_stats(
                np.mean(rwd_p2, axis=1))

        mat_, mal_ = [], []
        for i_s in range(n_subjs_):
            if ma_list[i_s] is not None:
                mat_.append(ma_list[i_s]['DM']['targ']['mu'][n_param:])
                mal_.append(ma_list[i_s]['DM']['lure']['mu'][n_param:])

        mat_mu[i], mat_se[i] = compute_stats(np.mean(mat_, axis=1))
        mal_mu[i], mal_se[i] = compute_stats(np.mean(mal_, axis=1))
        mad_mu[i], mad_se[i] = compute_stats(
            np.mean(mat_, axis=1) - np.mean(mal_, axis=1))

    # rwd_gm = rwd_mu['RM'] + rwd_mu['DM'] + rwd_mu['NM']
    # # f, ax = plt.subplots(1, 1, figsize=(5, 4))
    # ax.plot(rwd_gm)
    # ax.set_xlabel('cmpt')
    # ax.set_ylabel('R')
    # ax.set_xticks(range(len(comp_vals)))
    # ax.set_xticklabels(comp_vals)
    # sns.despine()

    # # f, ax = plt.subplots(1, 1, figsize=(5, 4))
    # ax.errorbar(range(len(comp_vals)), mat_mu, mat_se)
    # # ax.errorbar(range(len(comp_vals)), mal_mu, mal_se)
    # ax.set_xlabel('cmpt')
    # ax.set_ylabel('ma')
    # ax.set_xticks(range(len(comp_vals)))
    # ax.set_xticklabels(comp_vals)
    # sns.despine()

    ax.errorbar(range(len(comp_vals)), mad_mu, mad_se, color=blues[pi])
    ax.legend(penaltys_test, title='penalty test',
              bbox_to_anchor=(1.05, 1.0), loc='upper left')

    ax.set_xlabel('Level of competition')
    ax.set_ylabel('targ - lure act.')
    ax.set_xticks(range(len(comp_vals)))
    ax.set_xticklabels(comp_vals)
    sns.despine()
ax.axhline(0, linestyle='--', color='grey')
f.tight_layout()
