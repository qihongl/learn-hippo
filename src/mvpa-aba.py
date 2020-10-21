import os
import torch
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from task import SequenceLearning
from analysis.neural import build_yob, build_cv_ids
from analysis.task import get_oq_keys
from utils.utils import chunk
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, get_test_data_dir, \
    pickle_load_dict, get_test_data_fname
from analysis import compute_cell_memory_similarity, compute_stats, \
    compute_n_trials_to_skip, trim_data, get_trial_cond_ids, process_cache


sns.set(style='white', palette='colorblind', context='poster')
cb_pal = sns.color_palette('colorblind')
alphas = [1 / 3, 2 / 3, 1]

log_root = '../log/'
exp_name = '0916-widesim-prandom'

seed = 0
supervised_epoch = 600
learning_rate = 7e-4

n_branch = 4
n_param = 16
enc_size = 16
n_event_remember_train = 2
def_prob = None

comp_val = .8
leak_val = 0

n_hidden = 194
n_hidden_dec = 128
eta = .1

penalty_random = 1
# testing param, ortho to the training directory
penalty_discrete = 1
penalty_onehot = 0
normalize_return = 1

# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0

# testing params
pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
n_examples_test = 256


'''loop over conditions for testing'''

epoch_load = 1600
penalty_train = 4
fix_cond = 'DM'

n_event_remember_test = 2
similarity_max_test = .9
similarity_min_test = 0
p_rm_ob = 0.4
n_events = 2
n_parts = 3
scramble = False
slience_recall_time = None
trunc = 8

# subj_id = 0
n_subjs = 10
T_TOTAL = n_events * n_parts * n_param


'''helper funcs'''


def separate_AB_data(data_split):
    '''given a list of data, e.g. [A1, B1, A2, B2, ...]
    return [A1, A2, ...], [B1, B2, ...]
    '''
    data_A = np.array([data_split[2 * i] for i in range(n_parts)])
    data_B = np.array([data_split[2 * i + 1] for i in range(n_parts)])
    return data_A, data_B


def fill_recalled_features(Y_ob_, rt=1):
    for ti in range(n_trials):
        for pi in np.arange(1, n_parts):
            for ppi in np.arange(pi):
                final_feature_vales = Y_ob_[ppi][ti, -1]
                for fi in range(n_param):
                    Y_ob_[pi][ti, rt:, fi][Y_ob_[pi][ti, rt:, fi]
                                           == -1] = final_feature_vales[fi]
    return Y_ob_


# prealloc
n_feats_decd_mu = np.zeros((n_subjs, n_parts, n_param))
scores = np.zeros((n_subjs, n_parts, n_param))

for i_s, subj_id in enumerate(range(n_subjs)):
    np.random.seed(subj_id)
    torch.manual_seed(subj_id)

    '''init'''
    p = P(
        exp_name=exp_name, sup_epoch=supervised_epoch,
        n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
        enc_size=enc_size, n_event_remember=n_event_remember_train,
        penalty=penalty_train, penalty_random=penalty_random,
        penalty_onehot=penalty_onehot, penalty_discrete=penalty_discrete,
        normalize_return=normalize_return,
        p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
        n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
        lr=learning_rate, eta=eta,
    )

    task = SequenceLearning(
        n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
        p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
        similarity_cap_lag=p.n_event_remember,
        similarity_max=similarity_max_test,
        similarity_min=similarity_min_test
    )
    # create logging dirs
    log_path, log_subpath = build_log_path(
        subj_id, p, log_root=log_root, mkdir=False)
    test_data_fname = get_test_data_fname(n_examples_test, fix_cond=fix_cond)
    log_data_path = os.path.join(
        log_subpath['data'], f'n_event_remember-{n_event_remember_test}',
        f'p_rm_ob-{p_rm_ob}', f'similarity_cap-{similarity_min_test}_{similarity_max_test}')
    fpath = os.path.join(log_data_path, test_data_fname)
    if not os.path.exists(fpath):
        print('DNE')
        continue

    test_data_dict = pickle_load_dict(fpath)
    results = test_data_dict['results']
    XY = test_data_dict['XY']

    [dist_a_, Y_, log_cache_, log_cond_] = results
    [X_raw, Y_raw] = XY

    activity, [inpt] = process_cache(log_cache_, T_TOTAL, p)
    [C, H, M, CM, DA, V] = activity

    n_conds = len(TZ_COND_DICT)
    n_examples_skip = compute_n_trials_to_skip(log_cond_, p)
    # n_examples = n_examples_test - n_examples_skip
    [dist_a, Y, log_cond, log_cache, X_raw, Y_raw, C, V, CM, inpt] = trim_data(
        n_examples_skip,
        [dist_a_, Y_, log_cond_, log_cache_, X_raw, Y_raw, C, V, CM, inpt]
    )
    # process the data
    n_trials, T_TOTAL, _ = np.shape(Y_raw)
    trial_id = np.arange(n_trials)
    cond_ids = get_trial_cond_ids(log_cond)

    '''analysis'''
    actions = np.argmax(dist_a_, axis=-1)
    targets = np.argmax(Y_, axis=-1)

    targets_splits = np.array_split(targets, n_parts * n_events, axis=1)
    targets_A = targets_splits[0]
    targets_B = targets_splits[1]

    '''MVPA for subject i '''
    # reformat data for MVPA analysis
    n_trials = np.shape(X_raw)[0]
    trial_id = np.arange(n_trials)
    o_keys = np.zeros((n_trials, T_TOTAL))
    o_vals = np.zeros((n_trials, T_TOTAL))
    for i in trial_id:
        o_keys[i], _, o_vals[i] = get_oq_keys(X_raw[i], task)
    o_keys_splits = np.array_split(o_keys, n_parts * n_events, axis=1)
    o_vals_splits = np.array_split(o_vals, n_parts * n_events, axis=1)
    o_keys_A, o_keys_B = separate_AB_data(o_keys_splits)
    o_vals_A, o_vals_B = separate_AB_data(o_vals_splits)
    Y_ob_A = np.array(
        [build_yob(o_keys_A[pi], o_vals_A[pi]) for pi in range(n_parts)]
    )
    Y_ob_B = np.array(
        [build_yob(o_keys_B[pi], o_vals_B[pi]) for pi in range(n_parts)]
    )
    rt = 0
    Y_ob_A = fill_recalled_features(Y_ob_A, rt)
    Y_ob_B = fill_recalled_features(Y_ob_B, rt)

    # constrct X
    CM_splits = np.array_split(CM, n_parts * n_events, axis=1)
    CM_A, CM_B = separate_AB_data(CM_splits)

    # start MVPA
    Y_ob_i = Y_ob_A
    CM_i = CM_A

    # 1. train on A1, B1 and test on A1, B1, leave x% trial out
    n_folds = 10
    # cvids = build_cv_ids(n_trials, n_folds)
    rc_alphas = np.logspace(-2, 5, num=4)
    parameters = {'C': rc_alphas}
    cvids = build_cv_ids(n_trials, n_folds)
    scores_0 = np.zeros((n_param, n_folds))
    Y_hat_0 = np.zeros((n_trials, n_param, n_param))
    Y_proba_0 = np.zeros((n_trials, n_param, n_param, n_branch + 1))
    # fi,ci=0,0
    for fi in range(n_param):
        for ci in range(n_folds):
            # collect the hidden state for all time points
            X_train = np.reshape(CM_i[0][cvids != ci, :, :], (-1, n_hidden))
            X_test = np.reshape(CM_i[0][cvids == ci, :, :], (-1, n_hidden))
            # collect the targets for feature fi
            Y_train = np.reshape(Y_ob_i[0][cvids != ci, :, fi], (-1,))
            Y_test = np.reshape(Y_ob_i[0][cvids == ci, :, fi], (-1,))
            # construct inner cv ids
            inner_cvids = build_cv_ids(len(X_train), n_folds - 1)
            cvgs = GridSearchCV(
                LogisticRegression(penalty='l2'), parameters,
                cv=PredefinedSplit(inner_cvids), return_train_score=True
            )
            cvgs.fit(X_train, Y_train)
            # fit the model: hidden state -> value for fi
            cvgs.best_estimator_.fit(X_train, Y_train)
            # test the best model
            scores_0[fi, ci] = cvgs.best_estimator_.score(X_test, Y_test)
            Y_hat_ = cvgs.best_estimator_.predict(X_test)
            Y_hat_0[cvids == ci, :, fi] = np.reshape(
                Y_hat_, (np.sum(cvids == ci), n_param))

    # 2. train on all of A1, B1, and generalize to Ak, Bk, for k > 1
    scores_1 = np.zeros((n_param, ))
    scores_2 = np.zeros((n_param, ))
    Y_hat_1rs = np.zeros((n_trials * n_param, n_param))
    Y_hat_2rs = np.zeros((n_trials * n_param, n_param))
    Y_proba_1rs = np.zeros((n_trials * n_param, n_param, n_branch + 1))
    Y_proba_2rs = np.zeros((n_trials * n_param, n_param, n_branch + 1))
    for fi in range(n_param):
        # collect the hidden state for all time points
        X_train = np.reshape(CM_i[0], (-1, n_hidden))
        # collect the targets for feature fi
        Y_train = np.reshape(Y_ob_i[0][:, :, fi], (-1,))
        cvids = build_cv_ids(len(X_train), n_folds)
        cvgs = GridSearchCV(
            LogisticRegression(penalty='l2'), parameters,
            cv=PredefinedSplit(cvids), return_train_score=True
        )
        cvgs.fit(X_train, Y_train)
        # fit the model: hidden state -> value for fi
        cvgs.best_estimator_.fit(X_train, Y_train)
        # test the best model on A2
        X_test = np.reshape(CM_i[1], (-1, n_hidden))
        Y_test = np.reshape(Y_ob_i[1][:, :, fi], (-1,))
        scores_1[fi] = cvgs.best_estimator_.score(X_test, Y_test)
        Y_hat_1rs[:, fi] = cvgs.best_estimator_.predict(X_test)
        # test the best model on part A3
        X_test = np.reshape(CM_i[2], (-1, n_hidden))
        Y_test = np.reshape(Y_ob_i[2][:, :, fi], (-1,))
        scores_2[fi] = cvgs.best_estimator_.score(X_test, Y_test)
        Y_hat_2rs[:, fi] = cvgs.best_estimator_.predict(X_test)

    # collect classifier performance
    scores[i_s, :, :] = np.vstack(
        [np.mean(scores_0, axis=1), scores_1, scores_2])
    # np.shape(scores)
    # np.mean(scores,axis=2)

    # compute number of features recalled for part 1,2,3
    n_feats_decd_0 = np.sum(Y_hat_0 != -1, axis=2).T
    Y_hat_1 = np.reshape(Y_hat_1rs, (n_trials, n_param, n_param))
    n_feats_decd_1 = np.sum(Y_hat_1 != -1, axis=2).T
    Y_hat_2 = np.reshape(Y_hat_2rs, (n_trials, n_param, n_param))
    n_feats_decd_2 = np.sum(Y_hat_2 != -1, axis=2).T

    # plot the number of features recalled for part 1,2,3
    n_feats_decd = [n_feats_decd_0, n_feats_decd_1, n_feats_decd_2]
    for ii, n_feats_decd_i in enumerate(n_feats_decd):
        mu_, se_ = compute_stats(n_feats_decd_i.T)
        n_feats_decd_mu[i_s, ii] = mu_

    f, ax = plt.subplots(1, 1, figsize=(7, 5))
    for ii, n_feats_decd_i in enumerate(n_feats_decd):
        ax.errorbar(x=range(n_param), y=mu_, yerr=se_,
                    color=cb_pal[0], alpha=alphas[ii], label=f'Block {ii}')
    ax.set_title('Number of features decoded')
    ax.legend()
    ax.set_xlabel('Time')
    sns.despine()
    f.tight_layout()

    # ti = 14
    # # for ti in range(n_trials):
    # decodability_mat = np.vstack([Y_hat_0[ti], Y_hat_1[ti], Y_hat_2[ti]]).T
    # decodability_mat[decodability_mat >= 0] = 1
    # # decodability_mat[decodability_mat <0]=1
    # f, ax = plt.subplots(1, 1, figsize=(9, 4))
    # for pi in np.arange(1, n_parts):
    #     ax.axvline(pi * n_param - .5, linestyle='--', color='grey')
    # ax.imshow(decodability_mat, cmap='bone')
    # for pi in range(n_parts):
    #     for t, (k_t, o_t) in enumerate(zip(o_keys_A[pi][ti], o_vals_A[pi][ti])):
    #         if np.isnan(o_t):
    #             continue
    #         rect = patches.Rectangle(
    #             (pi * n_param + t - .5, k_t - .5), 1, 1,
    #             edgecolor='green', facecolor='none', linewidth=3
    #         )
    #         ax.add_patch(rect)
    # ax.set_title('Feature decodability over time')
    # ax.set_ylabel('Features')
    # ax.set_xlabel('Time')
    # ax.set_xticks(
    #     np.array([n_param * pi for pi in range(n_parts)]) + n_param / 2)
    # ax.set_xticklabels([f'Block {pi}' for pi in range(n_parts)])
    # f.tight_layout()


'''plot'''

n_feats_decd_mu_rm0 = np.delete(n_feats_decd_mu, 2, axis=0)
np.shape(n_feats_decd_mu_rm0)


n_feats_decd_mu_mu, n_feats_decd_mu_se = compute_stats(
    n_feats_decd_mu_rm0, axis=0)
f, ax = plt.subplots(1, 1, figsize=(7, 5))
for ii in range(n_parts):
    ax.errorbar(x=range(n_param), y=n_feats_decd_mu_mu[ii],
                yerr=n_feats_decd_mu_se[ii],
                color=cb_pal[0], alpha=alphas[ii], label=f'Block {ii}')
ax.set_title('Number of features decoded')
# ax.legend()
ax.set_xlabel('Time')
sns.despine()
f.tight_layout()

'''compute average accuracy'''

f, ax = plt.subplots(1, 1, figsize=(6, 5))
mean_scores = np.mean(scores, axis=0).T
for pi in range(n_parts):
    ax.plot(mean_scores[:, pi], color=cb_pal[0],
            alpha=alphas[pi], label=f'Block {ii}')
ax.set_xlabel('Time')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
ax.legend()
sns.despine()
f.tight_layout()
print(np.mean(mean_scores, axis=0))
