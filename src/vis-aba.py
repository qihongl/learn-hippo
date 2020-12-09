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
exp_name = 'vary-test-penalty'

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

n_subjs = 15
T_TOTAL = n_events * n_parts * n_param

'''helper funcs'''


def corrr_recall_isc_increment(recall_measure, rs_A, rs_B, trunc=8):
    '''comptue the correlation between recall signature at time t
    vs. isc increment (isc at t -> isc at t+1)

    trunc int: # of later time points removed - if we only care about time
    points at the beginning
    '''
    recall_isc_r_A = np.zeros(n_parts * n_events,)
    recall_isc_r_B = np.zeros(n_parts * n_events,)

    for pid in range(n_parts * n_events):
        t_pi = np.arange(pid * n_param, (pid + 1) * n_param)
        # print(t_pi)
        recall_isc_r_A[pid], _ = pearsonr(
            np.ravel(recall_measure[:, t_pi[1]:t_pi[-(1 + trunc)]]),
            np.ravel(np.diff(rs_A[:, t_pi[0]:t_pi[-(1 + trunc)]], axis=1))
        )
        recall_isc_r_B[pid], _ = pearsonr(
            np.ravel(recall_measure[:, t_pi[1]:t_pi[-(1 + trunc)]]),
            np.ravel(np.diff(rs_B[:, t_pi[0]:t_pi[-(1 + trunc)]], axis=1))
        )
    return (np.abs(recall_isc_r_A) + np.abs(recall_isc_r_B)) / 2


def separate_AB_data(data_split):
    '''given a list of data, e.g. [A1, B1, A2, B2, ...]
    return [A1, A2, ...], [B1, B2, ...]
    '''
    data_A = np.array([data_split[2 * i] for i in range(n_parts)])
    data_B = np.array([data_split[2 * i + 1] for i in range(n_parts)])
    return data_A, data_B


# prealloc
inpt_isc_r = np.zeros((n_subjs, n_events * n_parts))
ma_isc_r = np.zeros((n_subjs, n_events * n_parts))
ma_targ_isc_r = np.zeros((n_subjs, n_events * n_parts))
ma_lure_isc_r = np.zeros((n_subjs, n_events * n_parts))
rs_A_mu = np.zeros((n_subjs, T_TOTAL))
rs_B_mu = np.zeros((n_subjs, T_TOTAL))
rs_A_se = np.zeros((n_subjs, T_TOTAL))
rs_B_se = np.zeros((n_subjs, T_TOTAL))
n_feats_decd_mu = np.zeros((n_subjs, n_parts, n_param))
# CM_g = [None] * n_subjs
# Yob_g = [None] * n_subjs

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
        print(f'DNE: {fpath}')
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
    comp = np.full(np.shape(inpt), comp_val)
    leak = np.full(np.shape(inpt), leak_val)
    actions = np.argmax(dist_a_, axis=-1)
    targets = np.argmax(Y_, axis=-1)

    # compute performance
    corrects = targets == actions
    dks = actions == p.dk_id
    mistakes = np.logical_and(targets != actions, ~dks)
    rewards = corrects.astype(int) - mistakes.astype(int)

    sim_cos, sim_lca = compute_cell_memory_similarity(
        C, V, inpt, leak, comp)
    ma_targ = sim_lca[:, :, 0]
    ma_lure = sim_lca[:, :, 1]

    # compute some stats
    corrects_mu, corrects_se = compute_stats(corrects)
    mistakes_mu, mistakes_se = compute_stats(mistakes)
    dks_mu, dks_se = compute_stats(dks)
    rewards_mu, rewards_se = compute_stats(rewards)
    inpt_mu, inpt_se = compute_stats(inpt)
    ma_targ_mu, ma_targ_se = compute_stats(ma_targ)
    # inpt_mu, inpt_se = compute_stats(ma_targ)

    corrects_mu_splits = np.array_split(corrects_mu, n_parts * n_events)
    corrects_se_splits = np.array_split(corrects_se, n_parts * n_events)
    mistakes_mu_splits = np.array_split(mistakes_mu, n_parts * n_events)
    mistakes_se_splits = np.array_split(mistakes_se, n_parts * n_events)
    dks_mu_splits = np.array_split(dks_mu, n_parts * n_events)
    dks_se_splits = np.array_split(dks_se, n_parts * n_events)
    rewards_mu_splits = np.array_split(rewards_mu, n_parts * n_events)
    rewards_se_splits = np.array_split(rewards_se, n_parts * n_events)
    inpt_mu_splits = np.array_split(inpt_mu, n_parts * n_events)
    inpt_se_splits = np.array_split(inpt_se, n_parts * n_events)

    corrects_mu_bp = np.zeros((n_parts, n_param))
    corrects_se_bp = np.zeros((n_parts, n_param))
    mistakes_mu_bp = np.zeros((n_parts, n_param))
    mistakes_se_bp = np.zeros((n_parts, n_param))
    dks_mu_bp = np.zeros((n_parts, n_param))
    dks_se_bp = np.zeros((n_parts, n_param))
    rewards_mu_bp = np.zeros((n_parts, n_param))
    rewards_se_bp = np.zeros((n_parts, n_param))
    inpt_mu_bp = np.zeros((n_parts, n_param))
    inpt_se_bp = np.zeros((n_parts, n_param))

    for ii, i in enumerate(np.arange(0, n_parts * n_events, 2)):
        corrects_mu_bp[ii] = np.mean(
            corrects_mu_splits[i: i + n_events], axis=0)
        corrects_se_bp[ii] = np.mean(
            corrects_se_splits[i: i + n_events], axis=0)
        mistakes_mu_bp[ii] = np.mean(
            mistakes_mu_splits[i: i + n_events], axis=0)
        mistakes_se_bp[ii] = np.mean(
            mistakes_se_splits[i: i + n_events], axis=0)
        dks_mu_bp[ii] = np.mean(dks_mu_splits[i: i + n_events], axis=0)
        dks_se_bp[ii] = np.mean(dks_se_splits[i: i + n_events], axis=0)
        rewards_mu_bp[ii] = np.mean(rewards_mu_splits[i: i + n_events], axis=0)
        rewards_se_bp[ii] = np.mean(rewards_se_splits[i: i + n_events], axis=0)
        inpt_mu_bp[ii] = np.mean(inpt_mu_splits[i: i + n_events], axis=0)
        inpt_se_bp[ii] = np.mean(inpt_se_splits[i: i + n_events], axis=0)

    ''' compute schema pattern similarity'''
    rs_A = np.zeros((n_trials, T_TOTAL))
    rs_B = np.zeros((n_trials, T_TOTAL))

    for i in range(n_trials):
        np.shape(C)
        C_i_z = (C[i] - np.mean(C[i], axis=0)) / np.std(C[i], axis=0)
        C_i_splits = np.array(np.array_split(C_i_z, n_parts * n_events))

        C_i_A = C_i_splits[np.arange(0, n_parts * n_events, 2), :, :]
        C_i_B = C_i_splits[np.arange(0, n_parts * n_events, 2) + 1, :, :]
        C_i_A = np.reshape(C_i_A, newshape=(-1, n_hidden))
        C_i_B = np.reshape(C_i_B, newshape=(-1, n_hidden))
        sch_pat_i_A = np.mean(C_i_A, axis=0, keepdims=True)
        sch_pat_i_B = np.mean(C_i_B, axis=0, keepdims=True)

        rs_A[i] = np.squeeze(cosine_similarity(C[i], sch_pat_i_A))
        rs_B[i] = np.squeeze(cosine_similarity(C[i], sch_pat_i_B))

    rs_A_mu[i_s], rs_A_se[i_s] = compute_stats(rs_A)
    rs_B_mu[i_s], rs_B_se[i_s] = compute_stats(rs_B)

    '''compute correlation(input gate, isc increment)'''

    inpt_isc_r[i_s] = corrr_recall_isc_increment(inpt, rs_A, rs_B, trunc)
    ma_targ_isc_r[i_s] = corrr_recall_isc_increment(
        ma_targ, rs_A, rs_B, trunc)
    ma_isc_r[i_s] = corrr_recall_isc_increment(
        (ma_targ + ma_lure) / 2, rs_A, rs_B)


'''plot'''

grey_pal = sns.color_palette('Greys', n_colors=n_parts)
lines = [Line2D([0], [0], color=c, linewidth=3) for c in grey_pal]
labels = ['Block %d' % i for i in range(n_parts)]
# accuracy, dks, mistakes - line plot
b_pals = sns.color_palette('Blues', n_colors=n_parts)
g_pals = sns.color_palette('Greens', n_colors=n_parts)
r_pals = sns.color_palette('Reds', n_colors=n_parts)
grey_pal = sns.color_palette('Greys', n_colors=n_parts)

f, axes = plt.subplots(1, 3, figsize=(16, 4))
for ii, i in enumerate(np.arange(0, n_parts * n_events, 2)):
    axes[0].errorbar(
        x=range(n_param), y=corrects_mu_bp[ii], yerr=corrects_se_bp[ii],
        color=b_pals[ii], label=f'{ii}'
    )
    axes[1].errorbar(
        x=range(n_param), y=dks_mu_bp[ii], yerr=dks_se_bp[ii],
        color=grey_pal[ii], label=f'{ii}',
    )
    axes[2].errorbar(
        x=range(n_param), y=mistakes_mu_bp[ii], yerr=mistakes_se_bp[ii],
        color=r_pals[ii], label=f'{ii}'
    )
for ax in axes:
    ax.set_xlabel('Time')
    ax.set_ylim([-.05, 1.05])
    # ax.legend(range(n_parts), title='Block ID')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axes[2].legend(
    lines, labels, ncol=1,
    # loc='upper center',
    # bbox_to_anchor=(0.5, 1.6)
)
axes[0].set_ylabel('Accuracy')
axes[1].set_ylabel('Don\'t knows')
axes[2].set_ylabel('Mistakes')
sns.despine()
f.tight_layout()

fname = f'../figs/{exp_name}/simulated-behav-chang-etal-2020.png'
f.savefig(fname, dpi=100, bbox_to_anchor='tight')

# lca params - line plot
b_pals = sns.color_palette('Blues', n_colors=n_parts)
f, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.errorbar(
    x=range(n_param), y=inpt_mu_bp[ii], yerr=inpt_se_bp[ii],
    color=b_pals[ii], label=f'{ii}'
)
ax.set_xlabel('Time')
ax.legend(range(n_parts), title='Block ID')
ax.set_ylim([-.05, 1.05])
ax.set_ylabel('Input gate')
sns.despine()
f.tight_layout()


'''plot - the switching effect'''
np.shape(rs_A_mu)
n_se = 1
rs_A_mumu, rs_A_muse = compute_stats(rs_A_mu, n_se=n_se)
rs_B_mumu, rs_B_muse = compute_stats(rs_B_mu, n_se=n_se)

f, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.axvline(n_param - 1, color='red', alpha=.3, linestyle='--')
ax.errorbar(x=range(T_TOTAL), y=rs_A_mumu, yerr=rs_A_muse)
ax.errorbar(x=range(T_TOTAL), y=rs_B_mumu, yerr=rs_B_muse)
ax.legend(
    ['event boundary', 'to typical A pattern', 'to typical B pattern'],
    # bbox_to_anchor=(0.5, 1.05)
)
ax.axhline(0, color='grey', alpha=.3, linestyle='--')
for eb in np.arange(0, T_TOTAL, n_param)[1:] - 1:
    ax.axvline(eb, color='red', alpha=.3, linestyle='--')
ax.set_xlabel('Time')
ax.set_ylabel('Pattern similarity')
sns.despine()
f.tight_layout()

'''plot - the switching effect - stacked'''
rs_A_mu_splits = np.array_split(rs_A_mumu, n_parts)
rs_B_mu_splits = np.array_split(rs_B_mumu, n_parts)
rs_A_se_splits = np.array_split(rs_A_muse, n_parts)
rs_B_se_splits = np.array_split(rs_B_muse, n_parts)

# sns.palplot(cb_pal)

cb_pal_br = [cb_pal[3], cb_pal[0]]

grey_pal = sns.color_palette('Greys', n_colors=n_parts)
lines = [Line2D([0], [0], color=c, linewidth=3) for c in grey_pal]
labels = ['Block %d' % i for i in range(n_parts)]
xticklabels = ['A', 'B']
lines += [Line2D([0], [0], color=c, linewidth=3) for c in cb_pal_br]
labels += ['to typical %s pattern' % ltr for ltr in xticklabels]
# lines += [Line2D([0], [0], color='black', linewidth=3, linestyle='--')]
# labels += ['event boundary']

f, ax = plt.subplots(1, 1, figsize=(8, 10))
ax.axvline(n_param - trunc, color='k', alpha=1, linestyle='--')
ax.axhline(0, color='k', alpha=1, linestyle='--')
for i in np.arange(n_parts)[::-1]:
    ax.errorbar(
        x=range((n_param - trunc) * n_events),
        y=rs_B_mu_splits[i][trunc:-trunc],
        yerr=rs_B_se_splits[i][trunc:-trunc],
        color=cb_pal_br[0], alpha=alphas[i]
    )
    ax.errorbar(
        x=range((n_param - trunc) * n_events),
        y=rs_A_mu_splits[i][trunc:-trunc],
        yerr=rs_A_se_splits[i][trunc:-trunc],
        color=cb_pal_br[1], alpha=alphas[i]
    )


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(
    lines, labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.5)
)
ax.set_xticks(np.arange(0, (n_param - trunc) * 2 + 1, 8))
ax.set_xticklabels(
    np.arange(0, (n_param - trunc) * 2 + 1, 8) - (n_param - trunc)
)
ax.set_xlabel(
    '    B Segment        |         A Segment \n  Time from event sequence onset')
ax.set_ylabel('Pattern similarity')
sns.despine()
f.tight_layout()
fname = f'../figs/{exp_name}/simulated-ps-chang-etal-2020.png'
f.savefig(fname, dpi=90, bbox_to_anchor='tight')


# recall strength - overlay blocks
sim_cos, sim_lca = compute_cell_memory_similarity(
    C, V, inpt, leak, comp)

grey_pal = sns.color_palette('Greys', n_colors=n_parts)
lines = [Line2D([0], [0], color=c, linewidth=3) for c in grey_pal]
labels = ['Block %d' % i for i in range(n_parts)]
xticklabels = ['A', 'B']
lines += [Line2D([0], [0], color=c, linewidth=3) for c in cb_pal[:2]]
labels += xticklabels

f, ax = plt.subplots(1, 1, figsize=(8, 6))
sim_lca_mu, sim_lca_se = compute_stats(sim_lca)
for i in range(np.shape(sim_lca_mu)[1]):
    sim_lca_mu_splits_i = np.array_split(sim_lca_mu[:, i], n_parts)
    sim_lca_se_splits_i = np.array_split(sim_lca_se[:, i], n_parts)
    for j in range(n_parts):
        ax.errorbar(
            x=range(n_param * n_events), y=sim_lca_mu_splits_i[j],
            yerr=sim_lca_se_splits_i[j],
            color=cb_pal[i], alpha=alphas[j]
        )
ax.legend(lines, labels, ncol=2, bbox_to_anchor=(0.2, 1.05))
ax.axhline(0, color='grey', alpha=.3, linestyle='--')
ax.axvline(n_param - 1, color='red', alpha=.3, linestyle='--')
xticks = np.arange(n_param, n_param * n_events + 1, n_param) - n_param // 2
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Time')
ax.set_ylabel('Memory activation')
sns.despine()
f.tight_layout()


'''avearge AB to show the average correlation between recall and ISC increment'''


def average_AB(recall_isc_r, n_se=3):
    # take the average between block A and B
    recall_isc_r_byblock = np.zeros((n_subjs, n_parts))
    recall_isc_r_byblock[:, 0] = (recall_isc_r[:, 0] + recall_isc_r[:, 1]) / 2
    recall_isc_r_byblock[:, 1] = (recall_isc_r[:, 2] + recall_isc_r[:, 3]) / 2
    recall_isc_r_byblock[:, 2] = (recall_isc_r[:, 4] + recall_isc_r[:, 5]) / 2
    # remove crashed model (empty row)
    all_zero_row_id = np.sum(recall_isc_r_byblock == 0, axis=1) == n_parts
    recall_isc_r_byblock = recall_isc_r_byblock[~all_zero_row_id, :]
    recall_isc_r_byblock_mu, recall_isc_r_byblock_se = compute_stats(
        recall_isc_r_byblock, n_se=n_se)
    return recall_isc_r_byblock_mu, recall_isc_r_byblock_se


# average A and B
inpt_isc_r_byblock_mu, inpt_isc_r_byblock_se = average_AB(inpt_isc_r)
ma_isc_r_byblock_mu, ma_isc_r_byblock_se = average_AB(ma_isc_r)

# only look at the value of block one and two that involves EM
ma_isc_r_withem = ma_isc_r[:, 2:]
all_zero_row_id = np.sum(ma_isc_r_withem == 0, axis=1) == (
    n_parts - 1) * n_events
ma_isc_r_withem = ma_isc_r_withem[~all_zero_row_id, :]


ma_isc_r_withem_mu, ma_isc_r_withem_se = compute_stats(
    np.reshape(ma_isc_r_withem, (-1)), n_se=1)
ci = (ma_isc_r_withem_mu - ma_isc_r_withem_se * 1.96,
      ma_isc_r_withem_mu + ma_isc_r_withem_se * 1.96)
print(ma_isc_r_withem_mu, ma_isc_r_withem_se * 1.96)
print('the correlation between memory activation and isc increment:')
print(f'mu = {ma_isc_r_withem_mu}, 95% CI = ({ci[0]},{ci[1]})')

# plot input gate -> ISC increment
f, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.errorbar(x=range(n_parts), y=inpt_isc_r_byblock_mu,
            yerr=inpt_isc_r_byblock_se)
ax.axhline(0, linestyle='--', color='grey', alpha=.3)
ax.set_title('Input gate vs. shift towards the schematic pattern ')
ax.set_ylabel('Correlation')
ax.set_xlabel('Block id')
sns.despine()
f.tight_layout()

# plot memory activation -> ISC increment
f, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.errorbar(x=range(n_parts), y=ma_isc_r_byblock_mu,
            yerr=ma_isc_r_byblock_se)
ax.axhline(0, linestyle='--', color='grey', alpha=.3)
ax.set_title('Recall vs. shift towards the schematic pattern ')
ax.set_ylabel('Linear correlation')
ax.set_xlabel('Block id')
sns.despine()
f.tight_layout()
