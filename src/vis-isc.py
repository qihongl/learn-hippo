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
exp_name = 'encsize_fixed'
# exp_name = 'july9_v9'

subj_ids = np.arange(7)
n_subjs = len(subj_ids)
all_conds = ['RM', 'DM', 'NM']

CMs_dlist, DAs_dlist = {k: [] for k in all_conds}, {k: [] for k in all_conds}

for subj_id, fix_cond in product(subj_ids, all_conds):
    print(subj_id, fix_cond)

    # subj_id = 0
    penalty = 4
    supervised_epoch = 300
    epoch_load = 600
    # n_epoch = 500
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

    slience_recall_time = None
    # slience_recall_time = 2

    n_examples_test = 512

    np.random.seed(subj_id)
    # torch.manual_seed(subj_id)

    '''init'''
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
    # create logging dirs
    log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)

    test_data_dir, test_data_fname = get_test_data_dir(
        log_subpath, epoch_load, pad_len_test,
        slience_recall_time, n_examples_test)

    if fix_cond is not None:
        test_data_fname = fix_cond + test_data_fname

    fpath = os.path.join(test_data_dir, test_data_fname)
    test_data_dict = pickle_load_dict(fpath)

    results = test_data_dict['results']
    XY = test_data_dict['XY']

    [dist_a_, Y_, log_cache_, log_cond_] = results
    [X_raw, Y_raw] = XY

    # compute ground truth / objective uncertainty (delay phase removed)
    true_dk_wm_, true_dk_em_ = batch_compute_true_dk(X_raw, task)

    '''precompute some constants'''
    # figure out max n-time-steps across for all trials
    T_part = n_param + pad_len_test
    T_total = T_part * task.n_parts
    #
    n_conds = len(TZ_COND_DICT)
    memory_types = ['targ', 'lure']
    ts_predict = np.array([t % T_part >= pad_len_test for t in range(T_total)])

    '''organize results to analyzable form'''
    # skip examples untill EM is full
    n_examples_skip = n_event_remember
    n_examples = n_examples_test - n_examples_skip
    data_to_trim = [dist_a_, Y_, log_cond_,
                    log_cache_, true_dk_wm_, true_dk_em_]
    [dist_a, Y, log_cond, log_cache, true_dk_wm, true_dk_em] = trim_data(
        n_examples_skip, data_to_trim)
    # process the data
    cond_ids = get_trial_cond_ids(log_cond)
    activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
    [C, H, M, CM, DA, V] = activity_
    [inpt, leak, comp] = ctrl_param_

    CMs_dlist[fix_cond].append(CM)
    DAs_dlist[fix_cond].append(DA)

# type formatting
CMs_darray, DAs_darray = {}, {}
for cond in all_conds:
    CMs_darray[cond] = np.array(CMs_dlist[cond]).transpose((0, 3, 2, 1))
    DAs_darray[cond] = np.array(DAs_dlist[cond]).transpose((0, 3, 2, 1))
    print(f'np.shape(CMs_darray[{cond}]) = {np.shape(CMs_darray[cond])}')
    print(f'np.shape(DAs_darray[{cond}]) = {np.shape(DAs_darray[cond])}')


'''isc'''


from brainiak.funcalign.srm import SRM
dim_srm = 64
srm = SRM(features=dim_srm)

test_prop = .5
n_examples_tr = int(n_examples * (1-test_prop))
n_examples_te = int(n_examples * test_prop)

data = CMs_darray
# data = DAs_darray

_, nH, _, _ = np.shape(data[cond])

# split data
data_tr, data_te = {}, {}
for cond in all_conds:
    data_tr[cond] = data[cond][:, :, :, :n_examples_tr]
    data_te[cond] = data[cond][:, :, :, n_examples_tr:]

# fit training set
data_tr_unroll = np.concatenate(
    [data_tr[cond].reshape(n_subjs, nH, -1) for cond in all_conds],
    axis=2
)
srm.fit(data_tr_unroll)

# transform to the shared space
data_te_srm = {}
for cond in all_conds:
    data_te_srm[cond] = [
        srm.transform(data_te[cond][:, :, :, i])
        for i in range(n_examples_te)
    ]


# compute the inter-subject t-RSM for the i-th trial


def compute_bs_pattern_corr(data_te_i_srm):
    T_ = np.shape(data_te_i_srm)[-1]
    bs_trsa_i = []
    for i_s in range(n_subjs):
        for j_s in np.arange(i_s+1, n_subjs):
            # print(i_s, j_s)
            bs_trsa_i.append(
                np.corrcoef(
                    data_te_i_srm[i_s].T, data_te_i_srm[j_s].T)[T_:, :T_]
            )
    bs_trsa_i = np.array(bs_trsa_i)
    # extract diag: corr(t,t)
    bs_trsa_diag = np.array([np.diag(bs_trsa_i) for bs_trsa_i in bs_trsa_i])
    # average over subj i - subj j pairs
    bs_trsa_diag_mu = np.mean(bs_trsa_diag, axis=0)
    return bs_trsa_diag_mu


'''start plotting'''
n_se = 3


# '''Inter-subject pattern correlation, within each condition'''
# bs_trsa_diag_mu = {}
# for cond in all_conds:
#     # compute the t-t mean s
#     bs_trsa_diag_mu[cond] = np.array([
#         compute_bs_pattern_corr(data_te_i_srm)
#         for data_te_i_srm in data_te_srm[cond]
#     ])
#
# # compute stats
# itrsm_mu, itrsm_sd = {}, {}
# for cond in cond_ids.keys():
#     itrsm_mu[cond], itrsm_sd[cond] = compute_stats(
#         bs_trsa_diag_mu[cond], n_se=n_se)
# # plot
# f, ax = plt.subplots(1, 1, figsize=(8, 4))
# for i_c, cond in enumerate(cond_ids.keys()):
#     print(i_c, cond)
#     ax.errorbar(
#         x=range(T_total), y=itrsm_mu[cond], yerr=itrsm_sd[cond],
#         label=cond
#     )
# ax.axvline(T_part, color='red', linestyle='--', alpha=.5)
# ax.legend()
#
# ax.set_xlabel('Time')
# ax.set_ylabel('Linear Correlation')
# ax.set_title('Inter-subject pattern correlation')
# sns.despine()
# f.tight_layout()


'''Inter-subject pattern correlation, RM vs. cond'''


def compute_bs_bc_trsm(data_te_srm_rm_i, data_te_srm_xm_i):
    n_subj_, nH_, T_ = np.shape(data_te_srm_rm_i)
    bs_bc_trsm = []
    for i_s in range(n_subjs):
        j_s_list = set(range(n_subjs)).difference([i_s])
        for j_s in j_s_list:
            bs_bc_trsm.append(
                np.corrcoef(data_te_srm_rm_i[i_s].T,
                            data_te_srm_xm_i[j_s].T)[:T_, T_:]
            )
    return np.mean(bs_bc_trsm, axis=0)


def compute_bs_bc_isc(data_te_srm_rm_i, data_te_srm_xm_i, win_size=5):
    '''
    compute average isc acorss all subject pairs
    for the same trial, for two condition
    '''
    isc_mu = []
    for i_s in range(n_subjs):
        j_s_list = set(range(n_subjs)).difference([i_s])
        for j_s in j_s_list:
            # for subj i vs. subj j, compute isc over time
            isc_mu_ij = []
            for t in np.arange(T_part, T_total-win_size):
                isc_mu_ij.append(np.mean(np.corrcoef(
                    data_te_srm_rm_i[i_s][:, t: t+win_size],
                    data_te_srm_xm_i[j_s][:, t: t+win_size]
                )[dim_srm:, :dim_srm]))
            isc_mu_ij = np.array(isc_mu_ij)
            isc_mu.append(isc_mu_ij)
    return np.mean(isc_mu, axis=0)


win_size = 4
bs_bc_trsm_diag = {k: [] for k in all_conds}
bs_bc_isc = {k: [] for k in all_conds}
for cond in all_conds:
    # pick a trial
    for i in range(n_examples_te):
        data_te_srm_rm_i = data_te_srm['RM'][i]
        data_te_srm_xm_i = data_te_srm[cond][i]
        # for this trial, compute inter-subject inter-condition pattern corr
        bs_bc_trsm_c_i = compute_bs_bc_trsm(data_te_srm_rm_i, data_te_srm_xm_i)
        # only extract the diag entries (t to t)
        bs_bc_trsm_diag[cond].append(np.diag(bs_bc_trsm_c_i))
        # isc
        bs_bc_isc[cond].append(
            compute_bs_bc_isc(data_te_srm_rm_i, data_te_srm_xm_i, win_size)
        )


'''plot pattern corr '''
# compute stats
mu_, er_ = {}, {}
for cond in cond_ids.keys():
    mu_[cond], er_[cond] = compute_stats(bs_bc_trsm_diag[cond], n_se=n_se)

# plot
f, ax = plt.subplots(1, 1, figsize=(7, 4))
for i_c, cond in enumerate(cond_ids.keys()):
    print(i_c, cond)
    ax.errorbar(
        x=range(T_part), y=mu_[cond][T_part:], yerr=er_[cond][T_part:],
        label=cond
    )
ax.legend([f'RM-{cond}' for cond in all_conds])
# ax.axvline(T_part, color='red', linestyle='--', alpha=.5)

ax.set_xlabel('Time')
ax.set_ylabel('Linear Correlation')
ax.set_title('Spatial inter-subject correlation')
sns.despine()
f.tight_layout()


'''plot pattern isc'''
# compute stats
mu_, er_ = {}, {}
for cond in cond_ids.keys():
    mu_[cond], er_[cond] = compute_stats(bs_bc_isc[cond], n_se=n_se)

# plot
f, ax = plt.subplots(1, 1, figsize=(6, 4))
for i_c, cond in enumerate(cond_ids.keys()):
    print(i_c, cond)
    ax.errorbar(
        x=range(len(mu_[cond])), y=mu_[cond], yerr=er_[cond],
        label=cond
    )
ax.legend([f'RM-{cond}' for cond in all_conds])
# ax.axvline(T_part, color='red', linestyle='--', alpha=.5)

ax.set_xlabel(f'Time (sliding window size = {win_size})')
ax.set_ylabel('Linear Correlation')
ax.set_title('Inter-subject correlation')
sns.despine()
f.tight_layout()


# f, axes = plt.subplots(3, 1, figsize=(8, 10))
# for i_c, cond in enumerate(cond_ids.keys()):
#     axes[i_c].plot(np.array(bs_bc_isc[cond]).T)
