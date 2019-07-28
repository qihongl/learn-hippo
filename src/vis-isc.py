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
    get_qsource, trim_data

from vis import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds,\
    plot_time_course_for_all_conds
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

log_root = '../log/'
exp_name = 'metalearn-penalty'

subj_ids = np.arange(6)
n_subjs = len(subj_ids)
all_conds = ['RM', 'DM', 'NM']

CMs_dlist, DAs_dlist = {k: [] for k in all_conds}, {k: [] for k in all_conds}

for subj_id, fix_cond in product(subj_ids, all_conds):
    print(f'subj_id = {subj_id}, cond = {fix_cond}')

    supervised_epoch = 600
    epoch_load = 900

    n_param = 16
    n_branch = 4
    enc_size = 16
    n_event_remember = 4

    n_hidden = 194
    n_hidden_dec = 128
    learning_rate = 5e-4
    eta = .1

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
    penalty_test = 2

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
        penalty=penalty_train,
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
    ts_predict = np.array([t % T_part >= pad_len_test for t in range(T_total)])

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

# type formatting
CMs_darray, DAs_darray = {}, {}
for cond in all_conds:
    CMs_darray[cond] = np.array(CMs_dlist[cond]).transpose((0, 3, 2, 1))
    DAs_darray[cond] = np.array(DAs_dlist[cond]).transpose((0, 3, 2, 1))
    print(f'np.shape(CMs_darray[{cond}]) = {np.shape(CMs_darray[cond])}')
    print(f'np.shape(DAs_darray[{cond}]) = {np.shape(DAs_darray[cond])}')


'''isc'''


from brainiak.funcalign.srm import SRM
# from sklearn.preprocessing import StandardScaler
dim_srm = 32
srm = SRM(features=dim_srm)

test_prop = .5
n_examples_tr = int(n_examples * (1-test_prop))
n_examples_te = int(n_examples * test_prop)

data = CMs_darray
data = DAs_darray

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
# mean centering
for i_s in range(n_subjs):
    mu_ = np.mean(data_tr_unroll[i_s], axis=1, keepdims=True)
    data_tr_unroll[i_s] = data_tr_unroll[i_s] - mu_
srm.fit(data_tr_unroll)

np.shape(data_tr_unroll[0])
np.mean(data_tr_unroll[0], axis=1)
np.mean(data_tr_unroll[0], axis=0)


data_te_unroll = np.concatenate(
    [data_te[cond].reshape(n_subjs, nH, -1) for cond in all_conds],
    axis=2
)

# compute subject specific intercept
intercepts = [
    np.mean(data_te_unroll[i_s], axis=1, keepdims=True)
    for i_s in range(n_subjs)
]

# transform to the shared space
data_te_srm = {}
for cond in all_conds:
    d_ = [data_te[cond][:, :, :, i] for i in range(n_examples_te)]
    d_centered = [d_[i] - intercepts for i in range(n_examples_te)]
    data_te_srm[cond] = [
        srm.transform(d_centered[i])
        for i in range(n_examples_te)
    ]


# np.shape(data_te[cond][:, :, :, i])

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


# ref_cond = 'NM'
win_size = 5

bs_bc_trsm_diag = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
bs_bc_isc = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}

for i_rc, ref_cond in enumerate(all_conds):
    for i_c, cond in enumerate(all_conds):
        # pick a trial
        if i_c >= i_rc:
            for i in range(n_examples_te):
                # for this trial ...
                data_te_srm_rm_i = data_te_srm[ref_cond][i]
                data_te_srm_xm_i = data_te_srm[cond][i]
                # compute inter-subject inter-condition pattern corr
                bs_bc_trsm_c_i = compute_bs_bc_trsm(
                    data_te_srm_rm_i, data_te_srm_xm_i
                )
                # only extract the diag entries (t to t)
                bs_bc_trsm_diag[ref_cond][cond].append(np.diag(bs_bc_trsm_c_i))
                # isc
                bs_bc_isc[ref_cond][cond].append(
                    compute_bs_bc_isc(
                        data_te_srm_rm_i, data_te_srm_xm_i, win_size
                    )
                )


'''plot pattern corr '''
sns.palplot(sns.color_palette(n_colors=8))
c_pal = sns.color_palette(n_colors=8)
color_id_pick = [0, 1, 2, 3, 4, 7]
c_pal = [c_pal[color_id] for color_id in color_id_pick]

# compute stats
n_se = 3
mu_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
er_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
for ref_cond in cond_ids.keys():
    for cond in cond_ids.keys():
        mu_[ref_cond][cond], er_[ref_cond][cond] = compute_stats(
            bs_bc_trsm_diag[ref_cond][cond], n_se=n_se)

# plot
f, ax = plt.subplots(1, 1, figsize=(9, 5))
color_id = 0
i_rc, ref_cond = 0, 'RM'
for i_rc, ref_cond in enumerate(cond_ids.keys()):
    for i_c, cond in enumerate(cond_ids.keys()):
        if i_c >= i_rc:
            ax.errorbar(
                x=range(T_part),
                y=mu_[ref_cond][cond][T_part:],
                yerr=er_[ref_cond][cond][T_part:],
                label=f'{ref_cond}-{cond}', color=c_pal[color_id]
            )
            color_id += 1
ax.legend(bbox_to_anchor=(1, 1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel('Time')
ax.set_ylabel('Linear Correlation')
ax.set_title('Spatial inter-subject correlation')
sns.despine()
f.tight_layout()


'''plot pattern isc'''
n_se = 1
# compute stats
mu_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
er_ = {rcn: {cn: [] for cn in all_conds} for rcn in all_conds}
for ref_cond in cond_ids.keys():
    for cond in cond_ids.keys():
        mu_[ref_cond][cond], er_[ref_cond][cond] = compute_stats(
            bs_bc_isc[ref_cond][cond], n_se=n_se)

# plot
f, ax = plt.subplots(1, 1, figsize=(8, 5))
color_id = 0
i_rc, ref_cond = 0, 'RM'
for i_rc, ref_cond in enumerate(cond_ids.keys()):
    for i_c, cond in enumerate(cond_ids.keys()):
        print(i_c, cond)
        if i_c >= i_rc:
            ax.errorbar(
                x=range(len(mu_[ref_cond][cond])),
                y=mu_[ref_cond][cond], yerr=er_[ref_cond][cond],
                label=f'{ref_cond}-{cond}', color=c_pal[color_id]
            )
            color_id += 1
ax.legend(bbox_to_anchor=(1, 1))
# ax.axvline(T_part, color='red', linestyle='--', alpha=.5)
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel(f'Time (sliding window size = {win_size})')
ax.set_ylabel('Linear Correlation')
ax.set_title('Inter-subject correlation')
sns.despine()
f.tight_layout()


'''prediction isc change'''
np.shape(bs_bc_isc['RM']['DM'])
