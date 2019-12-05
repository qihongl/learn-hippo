import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from exp_tz import run_tz
from utils.params import P
# from utils.utils import to_sqnp, to_np, to_sqpth, to_pth
from utils.io import build_log_path, get_test_data_dir, \
    pickle_load_dict, get_test_data_fname, pickle_save_dict
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_cell_memory_similarity, create_sim_dict, \
    compute_event_similarity, batch_compute_true_dk, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip,\
    compute_cell_memory_similarity_stats, sep_by_qsource, prop_true, \
    get_qsource, trim_data, compute_roc, get_hist_info, remove_none

from vis import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds,\
    plot_time_course_for_all_conds
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA
from matplotlib.lines import Line2D

sns.set(style='white', palette='colorblind', context='poster')

log_root = '../log/'
exp_name = 'penalty-random-discrete'
# exp_name = 'penalty-random-discrete-highdp'

seed = 0
supervised_epoch = 600
learning_rate = 7e-4

n_branch = 4
n_param = 16
enc_size = 16
n_event_remember_train = 2
def_prob = None

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
p_rm_ob_rcl_load = .3

# testing params
pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
n_examples_test = 256


'''loop over conditions for testing'''

# subj_ids = np.arange(5)
epoch_load = 1200
penalty_train = 4
penalty_test = 4
fix_cond = 'DM'

n_event_remember_test = 2
similarity_max_test = .4
similarity_min_test = 0
p_rm_ob = 0.5
n_examples = 256
n_parts = 3
scramble = False
slience_recall_time = None


subj_id = 0
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
    raise ValueError()

test_data_dict = pickle_load_dict(fpath)
results = test_data_dict['results']
XY = test_data_dict['XY']

[dist_a_, Y_, log_cache_, log_cond_] = results
[X_raw, Y_raw] = XY

'''analysis'''

T_total = np.shape(Y_)[1]
activity, ctrl_param = process_cache(log_cache_, T_total, p)
[C, H, M, CM, DA, V] = activity
[inpt, leak, cmpt] = ctrl_param

actions = np.argmax(dist_a_, axis=-1)
targets = np.argmax(Y_, axis=-1)

# compute performance
corrects = targets == actions
dks = actions == p.dk_id
mistakes = np.logical_and(targets != actions, ~dks)
rewards = corrects.astype(int) - mistakes.astype(int)

corrects_mu, corrects_se = compute_stats(corrects)
mistakes_mu, mistakes_se = compute_stats(mistakes)
dks_mu, dks_se = compute_stats(dks)
rewards_mu, rewards_se = compute_stats(rewards)

inpt_mu, inpt_se = compute_stats(inpt)
cmpt_mu, cmpt_se = compute_stats(cmpt)


n_events = 2
corrects_mu_splits = np.array_split(corrects_mu, n_parts*n_events)
corrects_se_splits = np.array_split(corrects_se, n_parts*n_events)
mistakes_mu_splits = np.array_split(mistakes_mu, n_parts*n_events)
mistakes_se_splits = np.array_split(mistakes_se, n_parts*n_events)
dks_mu_splits = np.array_split(dks_mu, n_parts*n_events)
dks_se_splits = np.array_split(dks_se, n_parts*n_events)
rewards_mu_splits = np.array_split(rewards_mu, n_parts*n_events)
rewards_se_splits = np.array_split(rewards_se, n_parts*n_events)

inpt_mu_splits = np.array_split(inpt_mu, n_parts*n_events)
inpt_se_splits = np.array_split(inpt_se, n_parts*n_events)
cmpt_mu_splits = np.array_split(cmpt_mu, n_parts*n_events)
cmpt_se_splits = np.array_split(cmpt_se, n_parts*n_events)

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
cmpt_mu_bp = np.zeros((n_parts, n_param))
cmpt_se_bp = np.zeros((n_parts, n_param))


for ii, i in enumerate(np.arange(0, n_parts*n_events, 2)):
    corrects_mu_bp[ii] = np.mean(corrects_mu_splits[i: i+n_events], axis=0)
    corrects_se_bp[ii] = np.mean(corrects_se_splits[i: i+n_events], axis=0)
    mistakes_mu_bp[ii] = np.mean(mistakes_mu_splits[i: i+n_events], axis=0)
    mistakes_se_bp[ii] = np.mean(mistakes_se_splits[i: i+n_events], axis=0)
    dks_mu_bp[ii] = np.mean(dks_mu_splits[i: i+n_events], axis=0)
    dks_se_bp[ii] = np.mean(dks_se_splits[i: i+n_events], axis=0)
    rewards_mu_bp[ii] = np.mean(rewards_mu_splits[i: i+n_events], axis=0)
    rewards_se_bp[ii] = np.mean(rewards_se_splits[i: i+n_events], axis=0)

    inpt_mu_bp[ii] = np.mean(inpt_mu_splits[i: i+n_events], axis=0)
    inpt_se_bp[ii] = np.mean(inpt_se_splits[i: i+n_events], axis=0)
    cmpt_mu_bp[ii] = np.mean(cmpt_mu_splits[i: i+n_events], axis=0)
    cmpt_se_bp[ii] = np.mean(cmpt_se_splits[i: i+n_events], axis=0)

'''plot'''

# return - barplot
f, ax = plt.subplots(1, 1, figsize=(6, 5))
xticks = range(n_parts)
xticklabels = ['A/B %d' % i for i in range(n_parts)]
ax.bar(
    x=xticks,
    height=np.mean(rewards_mu_bp, axis=1),
    yerr=np.mean(rewards_se_bp, axis=1),
)
ax.set_xticks(xticks)
# ax.set_xticklabels(xticklabels)
ax.set_xlabel('Block ID')
ax.set_ylabel('Average return')
sns.despine()
f.tight_layout()

# accuracy, dks, mistakes - bar plot
f, axes = plt.subplots(3, 1, figsize=(6, 10))
xticks = range(n_parts)
xticklabels = ['A/B %d' % i for i in range(n_parts)]
axes[0].bar(
    x=xticks,
    height=np.mean(corrects_mu_bp, axis=1),
    yerr=np.mean(corrects_se_bp, axis=1),
)
axes[1].bar(
    x=xticks,
    height=np.mean(dks_mu_bp, axis=1),
    yerr=np.mean(dks_se_bp, axis=1),
)
axes[2].bar(
    x=xticks,
    height=np.mean(mistakes_mu_bp, axis=1),
    yerr=np.mean(mistakes_se_bp, axis=1),
)
axes[0].set_ylabel('Accuracy')
axes[1].set_ylabel('% dks')
axes[2].set_ylabel('% errors')
for ax in axes:
    ax.set_xlabel('Block ID')
    ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels)
sns.despine()
f.tight_layout()

# accuracy, dks, mistakes - line plot
b_pals = sns.color_palette('Blues', n_colors=n_parts)
g_pals = sns.color_palette('Greens', n_colors=n_parts)
r_pals = sns.color_palette('Reds', n_colors=n_parts)
f, axes = plt.subplots(3, 1, figsize=(7, 12))
for ii, i in enumerate(np.arange(0, n_parts*n_events, 2)):
    axes[0].errorbar(
        x=range(n_param), y=corrects_mu_bp[ii], yerr=corrects_se_bp[ii],
        color=b_pals[ii], label=f'{ii}'
    )
    axes[1].errorbar(
        x=range(n_param), y=dks_mu_bp[ii], yerr=dks_se_bp[ii],
        color=g_pals[ii], label=f'{ii}'
    )
    axes[2].errorbar(
        x=range(n_param), y=mistakes_mu_bp[ii], yerr=mistakes_se_bp[ii],
        color=r_pals[ii], label=f'{ii}'
    )
for ax in axes:
    ax.set_xlabel('Time')
    ax.legend(range(n_parts), title='Block ID')
    ax.set_ylim([-.05, 1.05])
    ax.axhline(1, color='grey', linestyle='--')
    ax.axhline(0, color='grey', linestyle='--')

axes[0].set_ylabel('Accuracy')
axes[1].set_ylabel('Don\'t knows')
axes[2].set_ylabel('Mistakes')
sns.despine()
f.tight_layout()


# lca param - bar plot
f, axes = plt.subplots(2, 1, figsize=(6, 9))
xticks = range(n_parts)
xticklabels = ['A/B %d' % i for i in range(n_parts)]

axes[0].bar(
    x=xticks,
    height=np.mean(inpt_mu_bp, axis=1),
    yerr=np.mean(inpt_se_bp, axis=1),
)
axes[0].set_xticks(xticks)
# ax.set_xticklabels(xticklabels)
axes[0].set_xlabel('Block ID')
axes[0].set_ylabel('Input gate')
axes[0].set_ylim([.06, None])

axes[1].bar(
    x=xticks,
    height=np.mean(cmpt_mu_bp, axis=1),
    yerr=np.mean(cmpt_se_bp, axis=1),
    # bottom=np.mean(cmpt_mu_bp)/2
)
axes[1].set_xticks(xticks)
# ax.set_xticklabels(xticklabels)
axes[1].set_xlabel('Block ID')
axes[1].set_ylabel('Competition')
axes[1].set_ylim([.6, None])
# axes[1].set_ylim([np.mean(cmpt_mu_bp)-.1, None])
sns.despine()
f.tight_layout()


# lca params - line plot
b_pals = sns.color_palette('Blues', n_colors=n_parts)
g_pals = sns.color_palette('Greens', n_colors=n_parts)
f, axes = plt.subplots(2, 1, figsize=(7, 9))
for ii, i in enumerate(np.arange(0, n_parts*n_events, 2)):
    axes[0].errorbar(
        x=range(n_param), y=inpt_mu_bp[ii], yerr=inpt_se_bp[ii],
        color=b_pals[ii], label=f'{ii}'
    )
    axes[1].errorbar(
        x=range(n_param), y=cmpt_mu_bp[ii], yerr=cmpt_se_bp[ii],
        color=g_pals[ii], label=f'{ii}'
    )
for ax in axes:
    ax.set_xlabel('Time')
    ax.legend(range(n_parts), title='Block ID')
    ax.set_ylim([-.05, 1.05])

axes[0].set_ylabel('Input gate')
axes[1].set_ylabel('Competition')
sns.despine()
f.tight_layout()

# schema pattern similarity
from sklearn.metrics.pairwise import cosine_similarity

rs_A = np.zeros((n_examples//2, T_total))
rs_B = np.zeros((n_examples//2, T_total))

np.shape(C[i])

i = 0
for i in range(n_examples//2):
    C_i_z = (C[i] - np.mean(C[i], axis=0)) / np.std(C[i], axis=0)
    # C_i_z = C[i]
    C_i_splits = np.array(np.array_split(C_i_z, n_parts*n_events))

    C_i_A = C_i_splits[np.arange(0, n_parts*n_events, 2), :, :]
    C_i_B = C_i_splits[np.arange(0, n_parts*n_events, 2)+1, :, :]
    C_i_A = np.reshape(C_i_A, newshape=(-1, n_hidden))
    C_i_B = np.reshape(C_i_B, newshape=(-1, n_hidden))
    sch_pat_i_A = np.mean(C_i_A, axis=0, keepdims=True)
    sch_pat_i_B = np.mean(C_i_B, axis=0, keepdims=True)

    rs_A[i] = np.squeeze(cosine_similarity(C[i], sch_pat_i_A))
    rs_B[i] = np.squeeze(cosine_similarity(C[i], sch_pat_i_B))

rs_A_mu, rs_A_se = compute_stats(rs_A)
rs_B_mu, rs_B_se = compute_stats(rs_B)

# f, ax = plt.subplots(1, 1, figsize=(12, 4))
# ax.axvline(n_param-1, color='red', alpha=.3, linestyle='--')
# ax.errorbar(x=range(T_total), y=rs_A_mu, yerr=rs_A_se)
# ax.errorbar(x=range(T_total), y=rs_B_mu, yerr=rs_B_se)
# ax.legend(
#     ['event boundary', 'to typical A pattern', 'to typical B pattern'],
#     # bbox_to_anchor=(0.5, 1.05)
# )
# ax.axhline(0, color='grey', alpha=.3, linestyle='--')
# for eb in np.arange(0, T_total, n_param)[1:]-1:
#     ax.axvline(eb, color='red', alpha=.3, linestyle='--')
# ax.set_xlabel('Time')
# ax.set_ylabel('Pattern similarity')
# sns.despine()
# f.tight_layout()

# the switching effect
rs_A_mu_splits = np.array_split(rs_A_mu, n_parts)
rs_B_mu_splits = np.array_split(rs_B_mu, n_parts)
rs_A_se_splits = np.array_split(rs_A_se, n_parts)
rs_B_se_splits = np.array_split(rs_B_se, n_parts)

cb_pal = sns.color_palette('colorblind')
alphas = [1/3, 2/3, 1]

grey_pal = sns.color_palette('Greys', n_colors=n_parts)
lines = [Line2D([0], [0], color=c, linewidth=3) for c in grey_pal]
labels = ['Block %d' % i for i in range(n_parts)]
xticklabels = ['A', 'B']
lines += [Line2D([0], [0], color=c, linewidth=3) for c in cb_pal[:2]]
labels += ['to typical %s pattern' % ltr for ltr in xticklabels]
lines += [Line2D([0], [0], color='red', linewidth=3, alpha=.3, linestyle='--')]
labels += ['event boundary']

f, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.axvline(n_param-1, color='red', alpha=.3, linestyle='--')
for i in np.arange(n_parts)[::-1]:
    ax.errorbar(
        x=range(n_param*n_events), y=rs_A_mu_splits[i], yerr=rs_A_se_splits[i],
        color=cb_pal[0], alpha=alphas[i]
    )
    ax.errorbar(
        x=range(n_param*n_events), y=rs_B_mu_splits[i], yerr=rs_B_se_splits[i],
        color=cb_pal[1], alpha=alphas[i]
    )
ax.legend(lines, labels, ncol=2,
          # bbox_to_anchor=(0.15, 1.05)
          )
ax.axhline(0, color='grey', alpha=.3, linestyle='--')
ax.set_xlabel('Time')
ax.set_ylabel('Pattern similarity')
sns.despine()
f.tight_layout()

# recall strength - overlay blocks
sim_cos, sim_lca = compute_cell_memory_similarity(
    C, V, inpt, leak, cmpt)

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
            x=range(n_param*n_events), y=sim_lca_mu_splits_i[j],
            yerr=sim_lca_se_splits_i[j],
            color=cb_pal[i], alpha=alphas[j]
        )
ax.legend(lines, labels, ncol=2, bbox_to_anchor=(0.2, 1.05))
ax.axhline(0, color='grey', alpha=.3, linestyle='--')
ax.axvline(n_param-1, color='red', alpha=.3, linestyle='--')
xticks = np.arange(n_param, n_param*n_events+1, n_param) - n_param//2
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Time')
ax.set_ylabel('Recall strength')
sns.despine()
f.tight_layout()
