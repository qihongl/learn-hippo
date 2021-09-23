import os
import torch
import pickle
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler

from task import SequenceLearning
from analysis.neural import build_yob, build_cv_ids
from analysis.task import get_oq_keys
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, pickle_load_dict, get_test_data_fname, \
    pickle_save_dict, get_test_data_dir
from analysis import compute_stats, compute_n_trials_to_skip, trim_data, \
    get_trial_cond_ids, process_cache
warnings.filterwarnings("ignore")
log_root = '../log'
fpath = 'data/decode-results.pkl'
sns.set(style='white', palette='colorblind', context='poster')

exp_name = 'vary-test-penalty'
penalty_train = 4
fix_penalty = 2
fix_cond = 'DM'
epoch_load = 1000
n_subjs = 14

n_branch = 4
n_param = 16
T = n_param * 2
enc_size = 16
def_prob = None
cmpt = .8
penalty_random = 1
supervised_epoch = 600
# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0

# testing params
pad_len_test = 0
n_examples_test = 256
slience_recall_time = None
test_params = [fix_penalty, pad_len_test, slience_recall_time]
# store params
p = P(
    exp_name=exp_name, sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
    enc_size=enc_size, penalty=penalty_train, penalty_random=penalty_random,
    cmpt=cmpt, p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
)
# init a dummy task
task = SequenceLearning(n_param=p.env.n_param, n_branch=p.env.n_branch)

dacc = np.zeros((n_subjs, n_param, T))
Yob_all, Yhat_all = [], []
o_keys_p1_all, o_keys_p2_all = [], []
for i_s in range(n_subjs):
    # create logging dirs
    np.random.seed(i_s)
    log_path, log_subpath = build_log_path(i_s, p, log_root, mkdir=False)
    test_data_fname = get_test_data_fname(n_examples_test, fix_cond=fix_cond)
    test_data_dir, _ = get_test_data_dir(log_subpath, epoch_load, test_params)
    fpath = os.path.join(test_data_dir, test_data_fname)
    if not os.path.exists(fpath):
        print('DNE')
        continue
    # load data
    test_data_dict = pickle_load_dict(fpath)
    [dist_a_, Y_, log_cache_, log_cond_] = test_data_dict['results']
    [X_raw, Y_raw] = test_data_dict['XY']
    [C, H, M, CM, DA, V], [inpt] = process_cache(log_cache_, T, p)
    n_examples_skip = compute_n_trials_to_skip(log_cond_, p)
    # trim data, wait until the model has 2EMs loaded
    [dist_a, Y, log_cond, log_cache, X_raw, Y_raw, C, V, CM, inpt] = trim_data(
        n_examples_skip,
        [dist_a_, Y_, log_cond_, log_cache_, X_raw, Y_raw, C, V, CM, inpt]
    )
    # process the data
    n_trials, _, _ = np.shape(Y_raw)
    trial_id = np.arange(n_trials)

    # 0 -> dk; 1,2,3,4 -> events
    actions = np.argmax(dist_a_, axis=-1)
    # 1,2,3,4 -> events (after shifting by 1)
    targets = np.argmax(Y_, axis=-1) + 1

    # build Yob
    o_keys, o_vals = np.zeros((n_trials, T)), np.zeros((n_trials, T))
    for i in trial_id:
        o_keys[i], _, o_vals[i] = get_oq_keys(X_raw[i], task)
    o_keys_p1, o_keys_p2 = o_keys[:, :n_param], o_keys[:, n_param:]
    o_vals_p1, o_vals_p2 = o_vals[:, :n_param], o_vals[:, n_param:]
    o_keys_p1_all.append(o_keys_p1)
    o_keys_p2_all.append(o_keys_p2)
    Yob_p1 = build_yob(o_keys_p1, o_vals_p1)
    Yob_p2 = build_yob(o_keys_p2, o_vals_p2)

    # estimate the recall time
    rt = np.argmax(inpt[:, n_param:], axis=1)
    Yob_p2_dm = np.zeros(np.shape(Yob_p2))
    for i in trial_id:
        # for the dm trials, assume all features are recalled
        Yob_p2_dm[i, :, :] = np.tile(Yob_p2[i, -1, :], [n_param, 1])
        # before recall time, work like part 1 (observation-based labeling)
        Yob_p2_dm[i, :rt[i], :] = Yob_p2[i, :rt[i], :]
    Yob = np.hstack([Yob_p1, Yob_p2_dm])

    '''analysis'''
    # print(f'X: # trials x # time points x # units: {np.shape(CM)}')
    # print(f'Y: # trials x # time points x # features: {np.shape(Yob)}')
    n_folds = 5
    lrc = np.logspace(-2, 10, num=6)
    cvids = build_cv_ids(n_trials, n_folds)
    scaler = StandardScaler()
    ridge_reg = LogisticRegression(
        penalty='l2', max_iter=1000, solver='lbfgs', multi_class='multinomial'
    )

    # prealloc
    Yhat = np.zeros(np.shape(Yob))

    for f in range(n_param):
        # choose the f-th feature
        Yf = Yob[:, :, f]
        for i in range(n_folds):
            # split train/test
            X_te = CM[cvids == i]
            X_tr = CM[cvids != i]
            Y_te = Yf[cvids == i]
            Y_tr = Yf[cvids != i]
            # inner cv
            icvids = build_cv_ids(len(Y_tr), n_folds - 1)
            gscv = GridSearchCV(
                ridge_reg, param_grid={'C': lrc}, cv=PredefinedSplit(icvids),
                return_train_score=True, refit=True
            )
            # reshape X train, Y train
            X_tr_rs = np.reshape(X_tr, (-1, np.shape(X_tr)[-1]))
            Y_tr_rs = np.reshape(Y_tr, (-1))
            X_te_rs = np.reshape(X_te, (-1, np.shape(X_te)[-1]))
            Y_te_rs = np.reshape(Y_te, (-1))
            # normalize
            X_tr_rs = scaler.fit_transform(X_tr_rs)
            X_te_rs = scaler.fit_transform(X_te_rs)
            # fit model
            gscv.fit(X_tr_rs, Y_tr_rs)
            # predict on the test set
            Y_hat_te_rs = gscv.best_estimator_.predict(X_te_rs)
            # store in the yhat matrix
            Yhat[cvids == i, :, f] = np.reshape(Y_hat_te_rs, np.shape(Y_te))
        # compute the decoding accuracy of feature f over time
        dacc[i_s, f, :] = np.mean(Yhat[:, :, f] == Yf, axis=0)
    Yob_all.append(Yob)
    Yhat_all.append(Yhat)

'''data io'''
# save data
data_dict = {
    'Yob_all': Yob_all, 'Yhat_all': Yhat_all,
    'o_keys_p1_all': o_keys_p1_all, 'o_keys_p2_all': o_keys_p2_all,
    # 'Yhat': Yhat, 'Yob': Yob
}
pickle_save_dict(data_dict, fpath)

# load data
data_dict = pickle_load_dict(fpath)
Yhat_all = np.array(data_dict['Yhat_all'])
Yob_all = np.array(data_dict['Yob_all'])
o_keys_p1_all = data_dict['o_keys_p1_all']
o_keys_p2_all = data_dict['o_keys_p2_all']

'''visualize the results'''
# print(f'overall acc: {np.mean(Y_match)}')
Y_match = Yhat_all == Yob_all
Y_pred_match = np.logical_and(np.logical_and(
    Yhat_all != -1, Yob_all != -1), Y_match)
np.shape(Yhat_all)

# compute the decoding accuracy OVER TIME, averaged across subjects
match_ovt_mu, match_ovt_se = compute_stats(
    np.mean(np.mean(Y_match, axis=1), axis=0), axis=1)
pmatch_ovt_mu, pmatch_ovt_se = compute_stats(
    np.mean(np.mean(Y_pred_match, axis=1), axis=0), axis=1)

f, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].errorbar(
    x=range(n_param), y=pmatch_ovt_mu[:n_param], xerr=pmatch_ovt_se[:n_param])
axes[0].errorbar(
    x=range(n_param), y=match_ovt_mu[:n_param], xerr=match_ovt_se[:n_param])
axes[1].errorbar(
    x=range(n_param), y=pmatch_ovt_mu[n_param:], xerr=pmatch_ovt_se[n_param:])
axes[1].errorbar(
    x=range(n_param), y=match_ovt_mu[n_param:], xerr=match_ovt_se[n_param:])

axes[0].set_xlabel('Part 1')
axes[1].set_xlabel('Part 2')
axes[0].set_ylabel('% correct decoding')
axes[1].legend(['event prediction only', 'including don\'t know'])
for ax in axes:
    ax.set_ylim([0, 1.05])
    ax.set_xticks([0, n_param])
sns.despine()
f.tight_layout()

# compute the decoding accuracy, averaged across subjects
dacc_mu, dacc_se = compute_stats(np.mean(np.mean(dacc, axis=-1), axis=-1))

# vertical plot
f, ax = plt.subplots(1, 1, figsize=(3, 4))
ax.bar(x=range(1), width=1, height=dacc_mu, yerr=dacc_se)
ax.set_ylabel('MVPA Acc. = %.2f' % (dacc_mu))
ax.axhline(1, linestyle='--', color='grey')
ax.set_xticks([])
sns.despine()
f.tight_layout()

# make decoding heatmap for several trials
i_s = 0
i = 5
o_keys_p1, o_keys_p2 = o_keys_p1_all[i_s], o_keys_p2_all[i_s]
for i in range(10):
    decode_hmap = Y_pred_match[i_s][i].T
    f, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    axes[0].imshow(decode_hmap[:, :n_param], cmap='bone')
    axes[1].imshow(decode_hmap[:, n_param:], cmap='bone')
    axes[0].set_xlabel('part 1')
    axes[1].set_xlabel('part 2')
    axes[0].set_ylabel('Feature')

    for t, (f_p1, f_p2) in enumerate(zip(o_keys_p1[i], o_keys_p2[i])):
        rect = patches.Rectangle(
            (t - .5, f_p1 - .5), 1, 1,
            edgecolor='green', facecolor='none', linewidth=3
        )
        axes[0].add_patch(rect)
        rect = patches.Rectangle(
            (t - .5, f_p2 - .5), 1, 1,
            edgecolor='green', facecolor='none', linewidth=3
        )
        axes[1].add_patch(rect)
    for ax in axes:
        ax.set_xticks([0, 15])
        ax.set_xticklabels([0, 15])
    f.tight_layout()
