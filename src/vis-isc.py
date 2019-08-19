import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product
from scipy.stats import pearsonr
from task import SequenceLearning
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, load_ckpt, pickle_load_dict, \
    get_test_data_dir, get_test_data_fname
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_trsm, compute_cell_memory_similarity, create_sim_dict, \
    compute_auc_over_time, compute_event_similarity, batch_compute_true_dk, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip,\
    compute_cell_memory_similarity_stats, sep_by_qsource, prop_true, \
    get_qsource, trim_data, make_df

from vis import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds,\
    plot_time_course_for_all_conds
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA
from itertools import combinations
from scipy.special import comb
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

log_root = '../log/'
exp_name = 'penalty-fixed-discrete-simple_'

subj_ids = np.arange(10)
n_subjs = len(subj_ids)
all_conds = ['RM', 'DM', 'NM']

# supervised_epoch = 600
# epoch_load = 900
# learning_rate = 5e-4
supervised_epoch = 300
epoch_load = 600
learning_rate = 1e-3

n_param = 16
n_branch = 3
enc_size = 16
n_event_remember = 2

n_hidden = 194
n_hidden_dec = 128
eta = .1

penalty_random = 0
# testing param, ortho to the training directory
penalty_discrete = 1
penalty_onehot = 0
normalize_return = 1

# loading params
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = .3
pad_len_load = -1
penalty_train = 4
# testing params
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
pad_len_test = 0
penalty_test = 4

slience_recall_time = None
# slience_recall_time = 2

n_examples_test = 256


def prealloc():
    return {cond: [] for cond in all_conds}


CMs_dlist = {cond: [] for cond in all_conds}
DAs_dlist = {cond: [] for cond in all_conds}

# C_dlist = {cond: None for cond in all_conds}
# V_dlist = {cond: None for cond in all_conds}
# inpt_dlist = {cond: None for cond in all_conds}
# leak_dlist = {cond: None for cond in all_conds}
# comp_dlist = {cond: None for cond in all_conds}

# cond_ids_dlist = {cond: None for cond in all_conds}
# cond_ids_combined = {cond: [] for cond in all_conds}

has_memory_conds = ['RM', 'DM']
ma_dlist = {cond: [] for cond in has_memory_conds}

# fix_cond = 'RM'

for subj_id in subj_ids:
    print(f'\nsubj_id = {subj_id}: ', end='')
    for fix_cond in all_conds:
        print(f'{fix_cond} ', end='')

        np.random.seed(subj_id)
        p = P(
            exp_name=exp_name, sup_epoch=supervised_epoch,
            n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
            enc_size=enc_size, n_event_remember=n_event_remember,
            penalty=penalty_train, penalty_random=penalty_random,
            penalty_onehot=penalty_onehot, penalty_discrete=penalty_discrete,
            normalize_return=normalize_return,
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
        log_path, log_subpath = build_log_path(
            subj_id, p, log_root=log_root, verbose=False
        )

        test_params = [penalty_test, pad_len_test, slience_recall_time]
        test_data_dir, test_data_subdir = get_test_data_dir(
            log_subpath, epoch_load, test_params)
        test_data_fname = get_test_data_fname(n_examples_test, fix_cond)
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
        ts_predict = np.array(
            [t % T_part >= pad_len_test for t in range(T_total)])

        '''organize results to analyzable form'''
        # skip examples untill EM is full
        n_examples_skip = n_event_remember
        n_examples = n_examples_test - n_examples_skip
        data_to_trim = [
            dist_a_, Y_, log_cond_, log_cache_, true_dk_wm_, true_dk_em_
        ]
        [dist_a, Y, log_cond, log_cache, true_dk_wm, true_dk_em] = trim_data(
            n_examples_skip, data_to_trim)
        # process the data
        cond_ids = get_trial_cond_ids(log_cond)
        activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
        [C, H, M, CM, DA, V] = activity_
        [inpt, leak, comp] = ctrl_param_

        # collect data
        CMs_dlist[fix_cond].append(CM)
        DAs_dlist[fix_cond].append(DA)

        if fix_cond in has_memory_conds:
            # collect memory activation for RM and DM sessions
            _, sim_lca = compute_cell_memory_similarity(
                C, V, inpt, leak, comp)
            sim_lca_dict = create_sim_dict(
                sim_lca, cond_ids, n_targ=p.n_segments)
            ma_dlist[fix_cond].append(sim_lca_dict[fix_cond])

print(f'n_subjs = {n_subjs}')

# organize target memory activation
tma = {cond: [] for cond in has_memory_conds}
for cond in has_memory_conds:
    # extract max target activation as the metric for recall
    tma[cond] = np.array([
        np.max(ma_dlist[cond][i_s]['targ'], axis=-1)
        for i_s in range(n_subjs)
    ]).transpose((0, 2, 1))
    print(f'np.shape(tma[{cond}]) = {np.shape(tma[cond])}')


# organize brain activity
CMs_darray, DAs_darray = {}, {}
for cond in all_conds:
    CMs_darray[cond] = np.array(CMs_dlist[cond]).transpose((0, 3, 2, 1))
    DAs_darray[cond] = np.array(DAs_dlist[cond]).transpose((0, 3, 2, 1))
    print(f'np.shape(CMs_darray[{cond}]) = {np.shape(CMs_darray[cond])}')
    print(f'np.shape(DAs_darray[{cond}]) = {np.shape(DAs_darray[cond])}')


'''isc'''

from brainiak.funcalign.srm import SRM
# from sklearn.preprocessing import StandardScaler
dim_srm = 16
srm = SRM(features=dim_srm)

test_prop = .5
n_examples_tr = int(n_examples * (1-test_prop))
n_examples_te = n_examples-n_examples_tr

# data = CMs_darray
data = DAs_darray

_, nH, _, _ = np.shape(data[cond])

# split data
data_tr = {cond: [] for cond in all_conds}
data_te = {cond: [] for cond in all_conds}

for cond in all_conds:
    data_tr_cond = data[cond][:, :, :, :n_examples_tr]
    data_te_cond = data[cond][:, :, :, n_examples_tr:]

    # mean centering
    for i_s in range(n_subjs):
        d_tr_i_s_ = data_tr_cond[i_s].reshape(nH, -1)
        d_te_i_s_ = data_te_cond[i_s].reshape(nH, -1)
        # mean center for each condition, for each subject
        mu_i_s_ = np.mean(d_tr_i_s_, axis=1, keepdims=True)
        data_tr[cond].append(d_tr_i_s_ - mu_i_s_)
        data_te[cond].append(d_te_i_s_ - mu_i_s_)


# fit training set
data_tr_unroll = np.concatenate(
    [data_tr[cond] for cond in all_conds],
    axis=2
)
# organize the data for srm form
data_tr_all_conds = np.moveaxis(
    [data_tr[cond] for cond in all_conds], source=0, destination=-1
).reshape(n_subjs, nH, -1)

data_te_all_conds = np.moveaxis(
    [data_te[cond] for cond in all_conds], source=0, destination=-1
).reshape(n_subjs, nH, -1)

srm.fit(data_tr_all_conds)
X_test_srm_ = srm.transform(data_te_all_conds)
X_test_srm_bycond = np.moveaxis(
    np.reshape(X_test_srm_, newshape=(n_subjs, dim_srm, -1, 3)),
    source=-1, destination=0
)

X_test_srm = {cond: None for cond in all_conds}
for i, cond in enumerate(all_conds):
    X_test_srm_cond_ = X_test_srm_bycond[i].reshape(
        n_subjs, dim_srm, -1, n_examples_te)
    X_test_srm[cond] = np.moveaxis(X_test_srm_cond_, source=-1, destination=0)


'''Inter-subject pattern correlation, RM vs. cond'''


def compute_bs_bc_trsm(data_te_srm_rm_i, data_te_srm_xm_i, return_mean=True):
    _, m_, n_ = np.shape(data_te_srm_rm_i)
    bs_bc_trsm = []
    # loop over subject i-j combinations
    for (i_s, j_s) in combinations(range(n_subjs), 2):
        bs_bc_trsm.append(
            np.corrcoef(
                data_te_srm_rm_i[i_s].T,
                data_te_srm_xm_i[j_s].T
            )[:n_, n_:]
        )
    if return_mean:
        return np.mean(bs_bc_trsm, axis=0)
    return bs_bc_trsm


def compute_bs_bc_isc(
    data_te_srm_rm_i, data_te_srm_xm_i,
    win_size=5, return_mean=True,
):
    '''
    compute average isc acorss all subject pairs
    for the same trial, for two condition
    '''
    isc_mu = []
    for (i_s, j_s) in combinations(range(n_subjs), 2):
        isc_mu_ij = []
        # compute sliding window averages
        for t in np.arange(T_part, T_total-win_size):
            isc_mu_ij.append(
                np.mean(np.diag(np.corrcoef(
                    data_te_srm_rm_i[i_s][:, t: t+win_size],
                    data_te_srm_xm_i[j_s][:, t: t+win_size]
                )[dim_srm:, :dim_srm]))
            )
        isc_mu_ij = np.array(isc_mu_ij)
        isc_mu.append(isc_mu_ij)
    if return_mean:
        return np.mean(isc_mu, axis=0)
    return isc_mu


# ref_cond = 'NM'
win_size = 5

bs_bc_sisc = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
bs_bc_tisc = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
bs_bc_sw_tisc = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}

for i_rc, ref_cond in enumerate(all_conds):
    for i_c, cond in enumerate(all_conds):
        # pick a trial
        if i_c >= i_rc:
            for i in range(n_examples_te):
                # for this trial ...
                data_te_srm_rm_i = X_test_srm[ref_cond][i]
                data_te_srm_xm_i = X_test_srm[cond][i]

                # compute inter-subject inter-condition pattern corr
                bs_bc_trsm_c_i = compute_bs_bc_trsm(
                    data_te_srm_rm_i, data_te_srm_xm_i,
                    return_mean=False
                )
                # only extract the diag entries (t to t)
                bs_bc_sisc[ref_cond][cond].append(
                    [np.diag(sisc_mat_ij) for sisc_mat_ij in bs_bc_trsm_c_i]
                )

                # compute isc
                bs_bc_tisc_c_i = compute_bs_bc_trsm(
                    np.transpose(data_te_srm_rm_i, axes=(0, 2, 1)),
                    np.transpose(data_te_srm_xm_i, axes=(0, 2, 1)),
                    return_mean=False
                )
                bs_bc_tisc[ref_cond][cond].append(
                    [np.diag(tisc_mat_ij) for tisc_mat_ij in bs_bc_tisc_c_i]
                )

                # sw-isc
                bs_bc_sw_tisc[ref_cond][cond].append(
                    compute_bs_bc_isc(
                        data_te_srm_rm_i, data_te_srm_xm_i, win_size,
                        return_mean=False
                    )
                )


'''plot spatial pattern isc '''
# sns.palplot(sns.color_palette(n_colors=8))
c_pal = sns.color_palette(n_colors=8)
color_id_pick = [0, 1, 2, 3, 4, 7]
c_pal = [c_pal[color_id] for color_id in color_id_pick]

# compute stats
n_se = 3
mu_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
er_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
for ref_cond in cond_ids.keys():
    for cond in cond_ids.keys():
        d_ = np.array(bs_bc_sisc[ref_cond][cond])
        if len(d_) > 0:
            mu_[ref_cond][cond], er_[ref_cond][cond] = compute_stats(
                np.mean(d_, axis=1),
                n_se=n_se
            )


# plot
f, ax = plt.subplots(1, 1, figsize=(7, 4))
color_id = 0
i_rc, ref_cond = 0, 'RM'
# for i_rc, ref_cond in enumerate(cond_ids.keys()):
for i_c, cond in enumerate(cond_ids.keys()):
    if i_c >= i_rc:
        ax.errorbar(
            x=range(T_part),
            y=mu_[ref_cond][cond][T_part:],
            yerr=er_[ref_cond][cond][T_part:],
            label=f'{ref_cond}-{cond}', color=c_pal[color_id]
        )
        color_id += 1

# ax.legend(bbox_to_anchor=(1, 1))
ax.legend()
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel('Time')
ax.set_ylabel('Linear Correlation')
ax.set_title('Spatial inter-subject correlation')
sns.despine()
f.tight_layout()


'''plot temporal isc'''

# compute stats
n_se = 3
mu_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
er_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
for ref_cond in cond_ids.keys():
    for cond in cond_ids.keys():
        d_ = np.array(bs_bc_tisc[ref_cond][cond])
        if len(d_) > 0:
            mu_[ref_cond][cond], er_[ref_cond][cond] = compute_stats(
                np.mean(d_, axis=1),
                n_se=n_se
            )

# plot
sort_id = np.argsort(mu_['RM']['RM'])[::-1]

f, ax = plt.subplots(1, 1, figsize=(9, 5))
color_id = 0
i_rc, ref_cond = 0, 'RM'
# for i_rc, ref_cond in enumerate(cond_ids.keys()):
for i_c, cond in enumerate(cond_ids.keys()):
    if i_c >= i_rc:
        ax.errorbar(
            x=range(dim_srm),
            y=mu_[ref_cond][cond][sort_id],
            yerr=er_[ref_cond][cond][sort_id],
            label=f'{ref_cond}-{cond}', color=c_pal[color_id]
        )
        color_id += 1

ax.legend(bbox_to_anchor=(1, 1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel('SRM components (sorted by RM-RM ISC value)')
ax.set_ylabel('Linear Correlation')
ax.set_title('Temporal inter-subject correlation')
sns.despine()
f.tight_layout()


'''plot temporal isc - sliding window'''
n_se = 3
# compute stats
mu_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
er_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
for ref_cond in cond_ids.keys():
    for cond in cond_ids.keys():
        d_ = bs_bc_sw_tisc[ref_cond][cond]
        if len(d_) > 0:
            mu_[ref_cond][cond], er_[ref_cond][cond] = compute_stats(
                np.mean(d_, axis=1), n_se=n_se)


# plot
f, ax = plt.subplots(1, 1, figsize=(6, 4))
color_id = 0
i_rc, ref_cond = 0, 'RM'
# for i_rc, ref_cond in enumerate(cond_ids.keys()):
for i_c, cond in enumerate(cond_ids.keys()):
    print(i_c, cond)
    if i_c >= i_rc:
        ax.errorbar(
            x=range(len(mu_[ref_cond][cond])),
            y=mu_[ref_cond][cond], yerr=er_[ref_cond][cond],
            label=f'{ref_cond}-{cond}', color=c_pal[color_id]
        )
        color_id += 1
# ax.legend(bbox_to_anchor=(1, 1))
ax.legend()
# ax.axvline(T_part, color='red', linestyle='--', alpha=.5)
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel(f'Time (sliding window size = {win_size})')
ax.set_ylabel('Linear Correlation')
ax.set_title('Temporal inter-subject correlation')
sns.despine()
f.tight_layout()


'''prediction isc change'''

# rm_dm_sisc_p2 = np.array(bs_bc_sisc['RM']['DM'])[:, T_part:].T
# mu_sisc, se_sisc = compute_stats(rm_dm_sisc_p2.T)
# rm_dm_tisc_p2 = np.array(bs_bc_sw_tisc['RM']['DM']).T
# mu_tisc, se_tisc = compute_stats(rm_dm_tisc_p2.T)

#
# f, axes = plt.subplots(2, 1, figsize=(7, 8))
# axes[0].plot(rm_dm_sisc_p2, color=c_pal[0], alpha=.05)
# axes[0].errorbar(
#     x=range(len(mu_sisc)), y=mu_sisc, yerr=se_sisc * n_se,
#     color='k'
# )
# axes[0].set_title('Spatial ISC')
# axes[0].set_xlabel(f'Time')
#
# axes[1].plot(rm_dm_tisc_p2, color=c_pal[2], alpha=.1)
# axes[1].errorbar(
#     x=range(len(mu_tisc)), y=mu_tisc, yerr=se_tisc * n_se,
#     color='k'
# )
# axes[1].set_title('Temporal ISC')
# axes[1].set_xlabel(f'Time (sliding window size = {win_size})')
# for ax in axes:
#     ax.set_ylabel('Linear Corr.')
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# f.tight_layout()
# sns.despine()


'''analyze isc change'''
n_tps = 9
n_subj_pairs = int(comb(n_subjs, 2))

# compute recall scores
tma_dm_p2_test = tma['DM'][:, T_part:, n_examples_tr:]
recall = np.mean(tma_dm_p2_test, axis=0)
# np.shape(tma_dm_p2_test)
# np.shape(recall)

# cond = 'RM'

r_val_sisc = {cond: np.zeros((n_subj_pairs, n_tps))
              for cond in has_memory_conds}
p_val_sisc = {cond: np.zeros((n_subj_pairs, n_tps))
              for cond in has_memory_conds}
r_val_tisc = {cond: np.zeros((n_subj_pairs, n_tps - win_size))
              for cond in has_memory_conds}
p_val_tisc = {cond: np.zeros((n_subj_pairs, n_tps - win_size))
              for cond in has_memory_conds}

r_mu_sisc = {cond: None for cond in has_memory_conds}
r_se_sisc = {cond: None for cond in has_memory_conds}
r_mu_tisc = {cond: None for cond in has_memory_conds}
r_se_tisc = {cond: None for cond in has_memory_conds}

for cond in has_memory_conds:

    rmdm_sisc = np.zeros((n_subj_pairs, n_examples_te, T_part))
    rmdm_tisc = np.zeros((n_subj_pairs, n_examples_te, T_part-win_size))

    for i in range(n_examples_te):
        # for this trial ...
        data_te_srm_rm_i = X_test_srm['RM'][i]
        data_te_srm_dm_i = X_test_srm[cond][i]
        # compute inter-subject inter-condition pattern corr
        rmdm_sisc_i = compute_bs_bc_trsm(
            data_te_srm_rm_i, data_te_srm_dm_i, return_mean=False
        )
        rmdm_sisc_i_diag = np.array([np.diag(mat) for mat in rmdm_sisc_i])
        rmdm_sisc[:, i, :] = rmdm_sisc_i_diag[:, T_part:]

        # isc
        rmdm_tisc[:, i, :] = compute_bs_bc_isc(
            data_te_srm_rm_i, data_te_srm_dm_i, win_size, return_mean=False
        )

    tma_dm_p2_test = tma[cond][:, T_part:, n_examples_tr:]
    recall = np.zeros((n_subj_pairs, n_examples_te, T_part))
    for i_comb, (i_s, j_s) in enumerate(combinations(range(n_subjs), 2)):
        recall_ij = tma_dm_p2_test[i_s] + tma_dm_p2_test[j_s] / 2
        recall[i_comb] = recall_ij.T

    for t in range(n_tps):
        sisc_change_t = rmdm_sisc[:, :, t+1]-rmdm_sisc[:, :, t]
        for i_comb in range(n_subj_pairs):
            r_val_sisc[cond][i_comb, t], p_val_sisc[cond][i_comb, t] = pearsonr(
                recall[i_comb, :, t], sisc_change_t[i_comb])

    for t in range(n_tps-win_size):
        tisc_change_t = rmdm_tisc[:, :, t+1]-rmdm_tisc[:, :, t]
        recall_win_t = np.mean(recall[:, :, t:t+win_size], axis=-1)
        for i_comb in range(n_subj_pairs):
            r_val_tisc[cond][i_comb, t], p_val_tisc[cond][i_comb, t] = pearsonr(
                recall_win_t[i_comb], tisc_change_t[i_comb])

        #
    r_mu_sisc[cond], r_se_sisc[cond] = compute_stats(r_val_sisc[cond])
    r_mu_tisc[cond], r_se_tisc[cond] = compute_stats(r_val_tisc[cond])

# np.shape(r_val_sisc[cond])

'''plot s-isc'''
f, ax = plt.subplots(1, 1, figsize=(7, 5))
for cond in has_memory_conds:
    ax.errorbar(
        x=range(len(r_mu_sisc[cond])),
        y=r_mu_sisc[cond], yerr=r_se_sisc[cond], label=f'RM-{cond}'
    )
ax.axhline(0, color='grey', linestyle='--')
ax.set_xlabel('Time')
ax.set_ylabel('Linear Corr.')
ax.set_title('Correlation: recall vs. spatial ISC change')
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.legend()
sns.despine()
f.tight_layout()


xticklabels = [f'RM-{cond}' for cond in has_memory_conds]
f, ax = plt.subplots(1, 1, figsize=(6, 4))
# sns.violinplot(data=[r_mu_sisc[cond] for cond in has_memory_conds])
sns.violinplot(data=[np.ravel(r_val_sisc[cond]) for cond in has_memory_conds])
# np.ravel(r_val_sisc[cond])
# sns.swarmplot(data=[r_mu_sisc[cond] for cond in has_memory_conds])
ax.axhline(0, color='grey', linestyle='--')
ax.set_xticks(range(len(xticklabels)))
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Condition')
ax.set_ylabel('Linear Correlation')
ax.set_title('Correlation: recall vs. spatial ISC change')
sns.despine()
f.tight_layout()

import dabest
import pandas as pd
# data_dict = r_mu_sisc
data_dict = {}
for cond in list(r_mu_sisc.keys()):
    data_dict[f'RM-{cond}'] = np.mean(r_val_sisc[cond], axis=-1)

df = make_df(data_dict)
iris_dabest = dabest.load(
    data=df, x="Condition", y="Value", idx=list(data_dict.keys())
)
iris_dabest.mean_diff.plot(
    swarm_label='Linear correlation', fig_size=(4, 3),
)


# xticklabels = [f'RM-{cond}' for cond in has_memory_conds]
# f, ax = plt.subplots(1, 1, figsize=(6, 4))
# sns.swarmplot(data=[np.concatenate(r_val_sisc[cond]) for cond in has_memory_conds])
# ax.axhline(0, color='grey', linestyle='--')
# ax.set_xticks(range(len(xticklabels)))
# ax.set_xticklabels(xticklabels)
# ax.set_xlabel('Condition')
# ax.set_ylabel('Linear Correlation')
# ax.set_title('Correlation between recall and ISC change')
# sns.despine()
# f.tight_layout()

'''plot t-isc'''
f, ax = plt.subplots(1, 1, figsize=(7, 5))
for cond in has_memory_conds:
    ax.errorbar(
        x=range(len(r_mu_tisc[cond])),
        y=r_mu_tisc[cond], yerr=r_se_tisc[cond], label=f'RM-{cond}'
    )
ax.axhline(0, color='grey', linestyle='--')
ax.set_xlabel('Time')
ax.set_ylabel('Linear Corr.')
ax.set_title('Correlation: recall vs. ISC change')
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.legend()
sns.despine()
f.tight_layout()


f, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.violinplot(data=[r_mu_tisc[cond] for cond in has_memory_conds])
# sns.swarmplot(data=[r_mu_sisc[cond] for cond in has_memory_conds])
ax.axhline(0, color='grey', linestyle='--')
ax.set_xticks(range(len(xticklabels)))
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Condition')
ax.set_ylabel('Linear Correlation')
ax.set_title('Correlation: recall vs. ISC change')
sns.despine()
f.tight_layout()


data_dict = {}
for cond in list(r_mu_sisc.keys()):
    data_dict[f'RM-{cond}'] = np.mean(r_val_tisc[cond], axis=-1)

df = make_df(data_dict)
iris_dabest = dabest.load(
    data=df, x="Condition", y="Value", idx=list(data_dict.keys())
)
iris_dabest.mean_diff.plot(
    swarm_label='Linear correlation', fig_size=(4, 3),
)
