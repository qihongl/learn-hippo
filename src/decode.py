import os
import torch
import pickle
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
log_root = '../log'
sns.set(style='white', palette='colorblind', context='poster')
# cb_pal = sns.color_palette('colorblind')
# alphas = [1 / 3, 2 / 3, 1]

exp_name = 'vary-test-penalty'
penalty_train = 4
fix_penalty = 4
fix_cond = 'DM'
epoch_load = 1000
n_subjs = 15

n_branch = 4
n_param = 16
enc_size = 16
def_prob = None
cmpt = .4
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

dacc = np.zeros((n_subjs, n_param))
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
    T = np.shape(Y_raw)[1]
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

    # prealloc
    Y_hat = np.zeros(np.shape(Yob))

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
                LogisticRegression(penalty='l2', max_iter=1000),
                param_grid={'C': lrc}, cv=PredefinedSplit(icvids),
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
            # plt.plot(gscv.cv_results_['mean_test_score'])
            # print(gscv.cv_results_['mean_test_score'])
            # print(np.mean(gscv.cv_results_['mean_test_score']))
            # predict on the test set
            Y_hat_te_rs = gscv.best_estimator_.predict(X_te_rs)
            # store in the yhat matrix
            Y_hat[cvids == i, :, f] = np.reshape(Y_hat_te_rs, np.shape(Y_te))

        dacc[i_s, f] = np.sum(Y_hat[:, :, f] == Yf) / np.size(Yf)

'''visualize the results'''
# compute the decoding accuracy
dacc_mu, _ = compute_stats(dacc)
dacc_mu, dacc_se = compute_stats(dacc_mu)

f, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.barh(y=range(1), height=1, width=dacc_mu, xerr=dacc_se)
ax.set_xlabel('Decoding accuracy = %.2f' % (dacc_mu))
ax.axvline(1, linestyle='--', color='grey')
ax.set_yticks([])
sns.despine()
f.tight_layout()

f, ax = plt.subplots(1, 1, figsize=(3, 4))
ax.bar(x=range(1), width=1, height=dacc_mu, yerr=dacc_se)
ax.set_ylabel('MVPA Acc. = %.2f' % (dacc_mu))
ax.axhline(1, linestyle='--', color='grey')
ax.set_xticks([])
sns.despine()
f.tight_layout()

# make a decoding heatmap
Y_match = Y_hat == Yob

for i in range(10):
    decode_hmap = np.logical_and(Y_match[i], Y_hat[i] != -1)

    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.imshow(decode_hmap.T, cmap='bone')
    ax.axvline(n_param - .5, linestyle='--', color='grey')
    ax.set_xlabel('Time')
    ax.set_ylabel('Feature')
    for t, (f_p1, f_p2) in enumerate(zip(o_keys_p1[i], o_keys_p2[i])):
        rect = patches.Rectangle(
            (t - .5, f_p1 - .5), 1, 1,
            edgecolor='green', facecolor='none', linewidth=3
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            (t - .5 + n_param, f_p2 - .5), 1, 1,
            edgecolor='green', facecolor='none', linewidth=3
        )
        ax.add_patch(rect)
    f.tight_layout()
