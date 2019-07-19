import os
# import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from models.LCALSTM_v9 import LCALSTM as Agent
# from models.LCALSTM_v9 import LCALSTM as Agent
# from models.LCALSTM_v9 import LCALSTM as Agent
# from models import LCALSTM as Agent
from itertools import product
from scipy.stats import pearsonr
from task import SequenceLearning
# from exp_tz import run_tz
from utils.params import P
# from utils.utils import to_sqnp, to_np, to_sqpth, to_pth
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, load_ckpt, get_test_data_dir, pickle_load_dict
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_trsm, compute_cell_memory_similarity, create_sim_dict, \
    compute_auc_over_time, compute_event_similarity, batch_compute_true_dk, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip,\
    compute_cell_memory_similarity_stats, sep_by_qsource, prop_true, \
    get_qsource, trim_data

from vis import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds,\
    plot_time_course_for_all_conds
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')
log_root = '../log/'

'''fixed params'''
exp_name = 'encsize_fixed'

penalty = 4
supervised_epoch = 300
epoch_load = 600
n_param = 16
n_branch = 4
enc_size = 16
n_event_remember = 4

n_hidden = 194
n_hidden_dec = 128
learning_rate = 1e-3
eta = .1

# loading params
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = .3
pad_len_load = -1
# testing params
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
pad_len_test = 0

n_examples_test = 512

'''vary params'''
# subj_ids = [0]
subj_ids = np.arange(7)
n_subjs = len(subj_ids)
slience_recall_times = [None] + [t for t in range(n_param)]

Ys = []
Yhats = []

# CMs_dlist, DAs_dlist = {k: [] for k in all_conds}, {k: [] for k in all_conds}

emb_acc_mu = np.zeros((n_subjs, n_param, n_param))
emb_mis_mu = np.zeros((n_subjs, n_param, n_param))
emb_dk_mu = np.zeros((n_subjs, n_param, n_param))

ctr_emb_acc_mu = np.zeros((n_subjs, n_param))
ctr_emb_mis_mu = np.zeros((n_subjs, n_param))
ctr_emb_dk_mu = np.zeros((n_subjs, n_param))

for subj_id, srt in product(subj_ids, slience_recall_times):
    print(f'\nsubj_id = {subj_id}, srt = {srt}')
    np.random.seed(subj_id)
    p = P(
        exp_name=exp_name, sup_epoch=supervised_epoch,
        n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
        enc_size=enc_size, n_event_remember=n_event_remember,
        penalty=penalty,
        p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
        n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
        lr=learning_rate, eta=eta,
    )
    # init env
    task = SequenceLearning(
        n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
        p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
    )
    # load the data
    log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)
    test_data_dir, test_data_fname = get_test_data_dir(
        log_subpath, epoch_load, pad_len_test, srt, n_examples_test)
    fpath = os.path.join(test_data_dir, test_data_fname)
    test_data_dict = pickle_load_dict(fpath)
    results = test_data_dict['results']
    XY = test_data_dict['XY']
    #
    [dist_a_, Y_, log_cache_, log_cond_] = results
    [X_raw, Y_raw] = XY
    # compute ground truth / objective uncertainty (delay phase removed)
    true_dk_wm_, true_dk_em_ = batch_compute_true_dk(X_raw, task)

    '''precompute some constants'''
    # figure out max n-time-steps across for all trials
    T_part = n_param + pad_len_test
    T_total = T_part * task.n_parts
    #
    # n_conds = len(TZ_COND_DICT)
    # memory_types = ['targ', 'lure']
    # ts_predict = np.array([t % T_part >= pad_len_test for t in range(T_total)])

    '''organize results to analyzable form'''
    # skip examples untill EM is full
    n_examples_skip = n_event_remember
    n_examples = n_examples_test - n_examples_skip
    data_to_trim = [
        dist_a_, Y_, log_cond_, log_cache_, true_dk_wm_, true_dk_em_]
    [dist_a, Y, log_cond, log_cache, true_dk_wm, true_dk_em] = trim_data(
        n_examples_skip, data_to_trim)
    # process the data
    cond_ids = get_trial_cond_ids(log_cond)
    # activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
    # [C, H, M, CM, DA, V] = activity_
    # [inpt, leak, comp] = ctrl_param_

    Yhats.append(dist_a)
    Ys.append(Y)
    # make sure Y for different sliencing time are the same
    assert np.all(Y == Ys[0])

    '''analysis'''
    n_se = 3

    # behavioral measure
    actions = np.argmax(dist_a, axis=-1)
    targets = np.argmax(Y, axis=-1)
    #
    corrects = targets == actions
    dks = actions == p.dk_id
    mistakes = np.logical_and(targets != actions, ~dks)
    # compute query source
    q_source_all_conds = get_qsource(true_dk_em, true_dk_wm, cond_ids, p)
    [q_source_rm_p2, q_source_dm_p2, q_source_nm_p2] = q_source_all_conds.values()

    # get a condition
    cond_name = 'DM'
    corrects_cond_p2 = corrects[cond_ids[cond_name], n_param:]
    dk_cond_p2 = dks[cond_ids[cond_name], n_param:]
    mistakes_cond_p2 = mistakes[cond_ids[cond_name], n_param:]
    acc_cond_p2_stats = sep_by_qsource(
        corrects_cond_p2, q_source_dm_p2, n_se=n_se)
    dk_cond_p2_stats = sep_by_qsource(
        dk_cond_p2, q_source_dm_p2, n_se=n_se)
    mistakes_cond_p2_stats = sep_by_qsource(
        mistakes_cond_p2, q_source_dm_p2, n_se=n_se)

    if srt is not None:
        emb_acc_mu[subj_id, srt] = acc_cond_p2_stats['EM only'][0]
        emb_mis_mu[subj_id, srt] = mistakes_cond_p2_stats['EM only'][0]
        emb_dk_mu[subj_id, srt] = dk_cond_p2_stats['EM only'][0]
    else:
        ctr_emb_acc_mu[subj_id] = acc_cond_p2_stats['EM only'][0]
        ctr_emb_mis_mu[subj_id] = mistakes_cond_p2_stats['EM only'][0]
        ctr_emb_dk_mu[subj_id] = dk_cond_p2_stats['EM only'][0]


# srt = 0
# for srt in range(n_param):
#     emb_acc_mu[subj_id, srt]

# sns.palplot(sns.color_palette(n_colors=5))
c_pal = sns.color_palette(n_colors=4)
c_pal = [c_pal[2], c_pal[0], c_pal[3]]
alpha = .3
n_se = 1
capsize = 2


data_pack = {
    'correct': [ctr_emb_acc_mu, emb_acc_mu],
    'uncertain': [ctr_emb_dk_mu, emb_dk_mu],
    'error': [ctr_emb_mis_mu, emb_mis_mu]
}
n_rows = len(data_pack)
n_cols = 2

f, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9))

for i, (ylab, data_pack_y) in enumerate(data_pack.items()):
    [ctr_emb_y_mu, emb_y_mu] = data_pack_y
    # compute stats
    sli_ = np.array([[emb_y_mu[subj_id, t][t] for t in range(n_param)]
                     for subj_id in subj_ids])
    ctr_mu_, ctr_er_ = compute_stats(ctr_emb_y_mu, n_se=n_se)
    sli_mu_, sli_er_ = compute_stats(sli_, n_se=n_se)
    diff_mu_, diff_er_ = compute_stats(ctr_emb_y_mu - sli_, n_se=n_se)

    axes[i, 0].errorbar(
        x=range(n_param), y=ctr_mu_, yerr=ctr_er_, label='control',
        color=c_pal[i], capsize=capsize
    )
    axes[i, 0].errorbar(
        x=range(n_param), y=sli_mu_, yerr=sli_er_, label='exp. (silencing)',
        color=c_pal[i], linestyle='--', capsize=capsize
    )
    axes[i, 0].fill_between(
        range(n_param), ctr_mu_, sli_mu_,
        alpha=alpha, color=c_pal[i]
    )
    axes[i, 0].legend()
    axes[i, 0].set_ylabel(f'P({ylab})')

    axes[i, 1].axhline(0, color='grey', linestyle='--')
    axes[i, 1].errorbar(
        x=range(n_param), y=diff_mu_, yerr=diff_er_,
        color=c_pal[i], linestyle='None', capsize=capsize
    )
    axes[i, 1].fill_between(
        range(n_param), np.zeros(n_param), diff_mu_, label='difference',
        alpha=alpha, color=c_pal[i]
    )
    # axes[i, 1].legend()
    axes[i, 1].set_ylabel('control - exp.')

for col_id in range(n_cols):
    axes[-1, col_id].set_xlabel('Time, recall phase')
    ylim_col_i = get_ylim_bonds(axes[:, col_id])
    for row_id in range(n_rows):
        axes[row_id, col_id].set_ylim(ylim_col_i)

# f.tight_layout()
sns.despine()
f.tight_layout(rect=[0, 0, 1, 0.9])
f.suptitle('The effect of silencing recall at time t', y=.95, fontsize=18)
f.savefig('../figs/silence-em.png', dpi=150)

'''the control performance plot'''

ctr_acc_mu_, ctr_acc_er_ = compute_stats(ctr_emb_acc_mu, n_se=n_se)
ctr_dk_mu_, ctr_acc_er_ = compute_stats(ctr_emb_dk_mu, n_se=n_se)

f, ax = plt.subplots(1, 1, figsize=(6, 3.5))
plot_pred_acc_rcl(
    ctr_acc_mu_, ctr_acc_er_,
    ctr_acc_mu_+ctr_dk_mu_,
    p, f, ax,
    title=f'EM-based prediction performance, {cond_name}',
    baseline_on=False, legend_on=True,
)
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('Accuracy')
f.tight_layout()
sns.despine()

f.savefig('../figs/em-pa.png', dpi=150)
#
#
# '''the % of contribution of immediate recall'''
#
# sli_ = np.array([[emb_acc_mu[subj_id, t][t] for t in range(n_param)]
#                  for subj_id in subj_ids])
# # diff_mu_, diff_er_ = compute_stats(ctr_emb_acc_mu - sli_, n_se=n_se)
# acc_mu_, acc_er_ = compute_stats(sli_, n_se=n_se)
# plt.plot(ctr_acc_mu_)
# plt.plot(acc_mu_)
#
# f, ax = plt.subplots(1, 1, figsize=(5, 4))
# ax.plot(acc_mu_/ctr_acc_mu_)
# #
