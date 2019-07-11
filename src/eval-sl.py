from scipy.stats import sem
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.LCALSTM_v2_1 import LCALSTM as Agent
# from models import LCALSTM as Agent
from task import SequenceLearning
from exp_tz import run_tz
from utils.params import P
from utils.utils import to_sqnp, to_np, to_sqpth, to_pth
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, load_ckpt
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_trsm, compute_cell_memory_similarity, create_sim_dict, \
    compute_auc_over_time, compute_event_similarity, batch_compute_true_dk
from plt_helper import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

log_root = '../log/'
# exp_name = 'pred-delay'
exp_name = 'july11_v2_1'

subj_id = 1
penalty = 4
supervised_epoch = 300
epoch_load = 600
# n_epoch = 500
n_param = 15
n_branch = 4
n_hidden = 194
learning_rate = 1e-3
eta = .1
n_mem = 4

p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = .3
pad_len_load = -1

p_test = .3
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
pad_len_test = 0

np.random.seed(subj_id)
torch.manual_seed(subj_id)

'''init'''
p = P(
    exp_name=exp_name, sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
    penalty=penalty,
    p_rm_ob_enc=p_rm_ob_enc_load,
    p_rm_ob_rcl=p_rm_ob_rcl_load,
    n_hidden=n_hidden, lr=learning_rate, eta=eta, n_mem=n_mem
)
# init env
task = SequenceLearning(
    p.env.n_param, p.env.n_branch,
    pad_len=pad_len_test,
    p_rm_ob_enc=p_rm_ob_enc_test,
    p_rm_ob_rcl=p_rm_ob_rcl_test,
)
# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)

# load the agent back
agent = Agent(task.x_dim, p.net.n_hidden, p.a_dim, dict_len=p.net.n_mem)
agent, optimizer = load_ckpt(epoch_load, log_subpath['ckpts'], agent)


# %%
'''eval'''
# training objective
n_examples_test = 666
[results, metrics, XY] = run_tz(
    agent, optimizer, task, p, n_examples_test,
    supervised=False, learning=False, get_data=True,
)
[dist_a_, Y_, log_cache_, log_cond_] = results
[X_raw, Y_raw] = XY


# compute ground truth / objective uncertainty (delay phase removed)
true_dk_wm_, true_dk_em_ = batch_compute_true_dk(X_raw, task)

# skip examples untill em buffer is full
non_nm_trials = np.where(log_cond_ != TZ_COND_DICT.inverse['NM'])[0]
n_examples_skip = non_nm_trials[n_mem+1]
n_examples = n_examples_test - n_examples_skip
# skip
dist_a = dist_a_[n_examples_skip:]
Y = Y_[n_examples_skip:]
log_cond = log_cond_[n_examples_skip:]
log_cache = log_cache_[n_examples_skip:]
true_dk_wm = true_dk_wm_[n_examples_skip:]
true_dk_em = true_dk_em_[n_examples_skip:]

# figure out max n-time-steps across for all trials
T_part = n_param + pad_len_test
T_total = T_part * task.n_parts

# %%
'''predefine/compute some constants'''
# precompute some constants
n_conds = len(TZ_COND_DICT)
memory_types = ['targ', 'lure']
ts_predict = np.array([t % T_part >= pad_len_test for t in range(T_total)])
# plot
alpha = .5
n_se = 3
# colors
gr_pal = sns.color_palette('colorblind')[2:4]
#
fig_dir = os.path.join(log_subpath['figs'], f'delay-{pad_len_test}')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)


# sns.palplot(gr_pal)


'''upack results'''
# compute trial ids
cond_ids = {}
for cn in list(TZ_COND_DICT.values()):
    cond_id_ = TZ_COND_DICT.inverse[cn]
    cond_ids[cn] = log_cond == cond_id_

# network internal reps
inpt = np.full((n_examples, T_total), np.nan)
inpt = np.full((n_examples, T_total), np.nan)
leak = np.full((n_examples, T_total), np.nan)
comp = np.full((n_examples, T_total), np.nan)
C = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
H = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
M = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
CM = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
DA = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
V = [None] * n_examples

for i in range(n_examples):
    for t in range(T_total):
        # unpack data for i,t
        [vector_signal, scalar_signal, misc] = log_cache[i][t]
        [inpt_it, leak_it, comp_it] = scalar_signal
        [h_t, m_t, cm_t, des_act_t, V_i] = misc
        # cache data to np array
        inpt[i, t] = to_sqnp(inpt_it)
        leak[i, t] = to_sqnp(leak_it)
        comp[i, t] = to_sqnp(comp_it)
        H[i, t, :] = to_sqnp(h_t)
        M[i, t, :] = to_sqnp(m_t)
        CM[i, t, :] = to_sqnp(cm_t)
        DA[i, t, :] = to_sqnp(des_act_t)
        V[i] = V_i

# compute cell state
C = CM - M

# onehot to int
actions = np.argmax(dist_a, axis=-1)
targets = np.argmax(Y, axis=-1)
# compute performance
dks = actions == p.dk_id
corrects = targets == actions
mistakes = np.logical_and(targets != actions, ~dks)


'''plot behavioral performance'''

for cn in list(TZ_COND_DICT.values()):
    Y_ = Y[cond_ids[cn], :]
    dist_a_ = dist_a[cond_ids[cn], :]
    # compute performance for this condition
    acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
    dk_mu = compute_dk(dist_a_)
    # plot
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_pred_acc_full(
        acc_mu, acc_er, acc_mu+dk_mu,
        [n_param], p,
        f, ax,
        title=f'Performance on the TZ task: {cn}',
    )
    fig_path = os.path.join(fig_dir, f'tz-acc-{cn}.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

'''compare LCA params across conditions'''


def plot_time_course_for_all_conds(
        matrix, ax,
        axis1_start=0, xlabel=None, ylabel=None, title=None,
        frameon=False, add_legend=True,
):
    for i, cond_name in enumerate(TZ_COND_DICT.values()):
        submatrix_ = matrix[cond_ids[cond_name], axis1_start:]
        M_, T_ = np.shape(submatrix_)
        mu_, er_ = compute_stats(submatrix_, axis=0, n_se=n_se)
        ax.errorbar(x=range(T_), y=mu_, yerr=er_, label=cond_name)
    if add_legend:
        ax.legend(frameon=frameon)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


f, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
plot_time_course_for_all_conds(
    inpt, axes[0], axis1_start=T_part,
    title='"need" for episodic memories', ylabel='input strength'
)
plot_time_course_for_all_conds(
    leak, axes[1], axis1_start=T_part,
    title='leakiness of the memories', ylabel='leak'
)
plot_time_course_for_all_conds(
    comp, axes[2], axis1_start=T_part,
    title='competition across memories', ylabel='competition'
)
axes[-1].set_xlabel('Time, recall phase')
axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
if pad_len_test > 0:
    for ax in axes:
        ax.axvline(pad_len_test, color='grey', linestyle='--')
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'tz-lca-param.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''compute cell-memory similarity / memory activation '''
# compute similarity between cell state vs. memories
sim_cos, sim_lca = compute_cell_memory_similarity(C, V, inpt, leak, comp)
sim_cos_dict = create_sim_dict(sim_cos, cond_ids, n_targ=1)
sim_lca_dict = create_sim_dict(sim_lca, cond_ids, n_targ=1)
# compute stats
sim_cos_stats = {cn: {'targ': {}, 'lure': {}} for cn in cond_ids.keys()}
sim_lca_stats = {cn: {'targ': {}, 'lure': {}} for cn in cond_ids.keys()}

# split by conditions x target/lure
for cond_name in ['DM', 'RM']:
    for memory_type in sim_cos_stats[cond_name].keys():
        sim_cos_stats[cond_name][memory_type]['mu'], \
            sim_cos_stats[cond_name][memory_type]['er'] = compute_stats(
            np.mean(sim_cos_dict[cond_name][memory_type], axis=-1))
        sim_lca_stats[cond_name][memory_type]['mu'], \
            sim_lca_stats[cond_name][memory_type]['er'] = compute_stats(
            np.mean(sim_lca_dict[cond_name][memory_type], axis=-1))

sim_cos_stats['NM']['lure']['mu'], \
    sim_cos_stats['NM']['lure']['er'] = compute_stats(
    np.mean(sim_cos_dict['NM']['lure'], axis=-1))
sim_lca_stats['NM']['lure']['mu'], \
    sim_lca_stats['NM']['lure']['er'] = compute_stats(
    np.mean(sim_lca_dict['NM']['lure'], axis=-1))

# plot target/lure activation for all conditions
# sim_plt_ = sim_cos_stats
sim_plt_ = sim_lca_stats
f, axes = plt.subplots(3, 1, figsize=(5, 8))
for i, c_name in enumerate(cond_ids.keys()):
    for m_type in memory_types:
        if m_type == 'targ' and c_name == 'NM':
            continue
        color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
        axes[i].errorbar(
            x=range(T_part),
            y=sim_plt_[c_name][m_type]['mu'][T_part:],
            yerr=sim_plt_[c_name][m_type]['er'][T_part:],
            label=f'{m_type}', color=color_
        )
        axes[i].set_title(c_name)
        axes[i].set_ylabel('Memory activation')
axes[0].legend()
axes[-1].set_xlabel('Time, recall phase')
# make all ylims the same
ylim_l, ylim_r = get_ylim_bonds(axes)
for i, ax in enumerate(axes):
    ax.set_ylim([ylim_l, ylim_r])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

if pad_len_test > 0:
    for ax in axes:
        ax.axvline(pad_len_test, color='grey', linestyle='--')
f.tight_layout()
sns.despine()
fig_path = os.path.join(fig_dir, f'tz-memact-lca.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

'''use the amount of errors/dk in the last k time steps to predict recall'''
# pick a condition
cond_name = 'DM'

targ_act_cond_p2 = np.max(sim_lca_dict[cond_name]['targ'][:, T_part:], axis=-1)
dk_cond_p2 = dks[cond_ids[cond_name], n_param:]
mistakes_cond_p2 = mistakes[cond_ids[cond_name], n_param:]
true_dk_em_cond_p2 = true_dk_em[cond_ids[cond_name], n_param:]
true_dk_wm_cond_p2 = true_dk_wm[cond_ids[cond_name], :]

# t s.t. maximal recall peak
t_recall_peak = np.argmax(np.mean(targ_act_cond_p2, axis=0))


# plot
# plot target memory activation profile, for all trials
f, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(targ_act_cond_p2.T, alpha=.05, color=gr_pal[0])
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('Memory activation')
ax.set_title(f'Target activation, {cond_name}')
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-targact-lca.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

# use previous uncertainty to predict memory activation
t_pick_max = 5
v_pal = sns.color_palette('viridis', n_colors=t_pick_max)
f, ax = plt.subplots(1, 1, figsize=(8, 4))
for t_pick_ in range(t_pick_max):
    # compute number of don't knows produced so far
    ndks_p2_b4recall = np.sum(dk_cond_p2[:, :t_pick_], axis=1)
    nvs = np.unique(ndks_p2_b4recall)
    ma_mu = np.zeros(len(nvs),)
    ma_er = np.zeros(len(nvs),)
    for i, val in enumerate(np.unique(ndks_p2_b4recall)):
        mem_act_recall_ndk = targ_act_cond_p2[ndks_p2_b4recall == val, t_pick_]
        ma_mu[i], ma_er[i] = compute_stats(mem_act_recall_ndk, n_se=1)
    ax.errorbar(x=nvs, y=ma_mu, yerr=ma_er, color=v_pal[t_pick_])

ax.legend(range(t_pick_max), bbox_to_anchor=(1.3, 1))
ax.set_title('Recall ~ subjective uncertainty')
ax.set_xlabel('# don\'t knows')
ax.set_ylabel('average recall peak')
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-targact-by-ndk.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

# compute prediction memory source
em_only_cond_p2 = np.logical_and(true_dk_wm_cond_p2, ~ true_dk_em_cond_p2)
wm_only_cond_p2 = np.logical_and(~true_dk_wm_cond_p2, true_dk_em_cond_p2)
neither_cond_p2 = np.logical_and(true_dk_wm_cond_p2, true_dk_em_cond_p2)
both_cond_p2 = np.logical_and(~true_dk_wm_cond_p2, ~true_dk_em_cond_p2)

n_ = np.shape(both_cond_p2)[0]

# show source prop
cnts_em_only_cond_p2 = np.sum(em_only_cond_p2, axis=0)/n_
cnts_wm_only_cond_p2 = np.sum(wm_only_cond_p2, axis=0)/n_
cnts_neither_cond_p2 = np.sum(neither_cond_p2, axis=0)/n_
cnts_both_cond_p2 = np.sum(both_cond_p2, axis=0)/n_

width = .5
f, ax = plt.subplots(1, 1, figsize=(7, 4))
p1 = ax.bar(range(n_param), cnts_em_only_cond_p2, label='EM', width=width)
p2 = ax.bar(range(n_param), cnts_neither_cond_p2, label='neither',
            bottom=cnts_em_only_cond_p2, width=width)
p3 = ax.bar(range(n_param), cnts_both_cond_p2, label='both',
            bottom=cnts_em_only_cond_p2 + cnts_neither_cond_p2, width=width)
ax.set_ylabel('Proportion (%)')
ax.set_xlabel('Time, recall phase')
ax.set_title(f'Where is q? ({cond_name})')
ax.legend()
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-q-source.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


# use CURRENT uncertainty to predict memory activation
ma_em_mu, ma_em_er = np.zeros(n_param,), np.zeros(n_param,)
ma_wm_mu, ma_wm_er = np.zeros(n_param,), np.zeros(n_param,)
ma_bo_mu, ma_bo_er = np.zeros(n_param,), np.zeros(n_param,)
ma_nt_mu, ma_nt_er = np.zeros(n_param,), np.zeros(n_param,)

# compute stats
# t = 0
np.shape(targ_act_cond_p2)
T_part
for t in range(n_param):
    ma_em_mu[t], ma_em_er[t] = compute_stats(
        targ_act_cond_p2[em_only_cond_p2[:, t], t+pad_len_test], n_se=n_se)
    ma_wm_mu[t], ma_wm_er[t] = compute_stats(
        targ_act_cond_p2[wm_only_cond_p2[:, t], t+pad_len_test], n_se=n_se)
    ma_bo_mu[t], ma_bo_er[t] = compute_stats(
        targ_act_cond_p2[both_cond_p2[:, t], t+pad_len_test], n_se=n_se)
    ma_nt_mu[t], ma_nt_er[t] = compute_stats(
        targ_act_cond_p2[neither_cond_p2[:, t], t+pad_len_test], n_se=n_se)

f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.errorbar(x=range(n_param), y=ma_em_mu, yerr=ma_em_er, label='EM only')
ax.errorbar(x=range(n_param), y=ma_nt_mu, yerr=ma_nt_er, label='neither')
ax.errorbar(x=range(n_param), y=ma_bo_mu, yerr=ma_bo_er, label='both')
if np.sum(wm_only_cond_p2) > 0:
    ax.errorbar(x=range(n_param), y=ma_wm_mu, yerr=ma_wm_er, label='WM only')

ax.set_title(f'Target memory activation, {cond_name}')
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('Activation')
ax.legend(fancybox=True)
f.tight_layout()
sns.despine()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-targact-by-dk.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''predictions performance w.r.t prediction source (EM, WM)'''

corrects_cond_p2 = corrects[cond_ids[cond_name], n_param:]

# prealloc
pa_em_mu, pa_em_er = np.zeros(n_param,), np.zeros(n_param,)
pa_wm_mu, pa_wm_er = np.zeros(n_param,), np.zeros(n_param,)
pa_bo_mu, pa_bo_er = np.zeros(n_param,), np.zeros(n_param,)
pa_nt_mu, pa_nt_er = np.zeros(n_param,), np.zeros(n_param,)
for t in range(n_param):
    pa_em_mu[t], pa_em_er[t] = compute_stats(
        corrects_cond_p2[em_only_cond_p2[:, t], t], n_se=n_se)
    pa_wm_mu[t], pa_wm_er[t] = compute_stats(
        corrects_cond_p2[wm_only_cond_p2[:, t], t], n_se=n_se)
    pa_bo_mu[t], pa_bo_er[t] = compute_stats(
        corrects_cond_p2[both_cond_p2[:, t], t], n_se=n_se)
    pa_nt_mu[t], pa_nt_er[t] = compute_stats(
        corrects_cond_p2[neither_cond_p2[:, t], t], n_se=n_se)

f, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.errorbar(x=range(n_param), y=pa_em_mu, yerr=pa_em_er, label='EM only')
ax.errorbar(x=range(n_param), y=pa_nt_mu, yerr=pa_nt_er, label='neither')
ax.errorbar(x=range(n_param), y=pa_bo_mu, yerr=pa_bo_er, label='both')
if np.sum(wm_only_cond_p2) > 0:
    ax.errorbar(x=range(n_param), y=pa_wm_mu, yerr=pa_wm_er, label='WM only')
ax.set_title(f'Performance, {cond_name}')
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('Accuracy')
ax.legend(fancybox=True)
f.tight_layout()
sns.despine()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-pa-by-dk.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''dk w.r.t prediction source (EM, WM)'''

# prealloc
dk_em_mu, dk_em_er = np.zeros(n_param,), np.zeros(n_param,)
dk_wm_mu, dk_wm_er = np.zeros(n_param,), np.zeros(n_param,)
dk_bo_mu, dk_bo_er = np.zeros(n_param,), np.zeros(n_param,)
dk_nt_mu, dk_nt_er = np.zeros(n_param,), np.zeros(n_param,)
for t in range(n_param):
    dk_em_mu[t], dk_em_er[t] = compute_stats(
        dk_cond_p2[em_only_cond_p2[:, t], t], n_se=n_se)
    dk_wm_mu[t], dk_wm_er[t] = compute_stats(
        dk_cond_p2[wm_only_cond_p2[:, t], t], n_se=n_se)
    dk_bo_mu[t], dk_bo_er[t] = compute_stats(
        dk_cond_p2[both_cond_p2[:, t], t], n_se=n_se)
    dk_nt_mu[t], dk_nt_er[t] = compute_stats(
        dk_cond_p2[neither_cond_p2[:, t], t], n_se=n_se)

f, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.errorbar(x=range(n_param), y=dk_em_mu, yerr=dk_em_er, label='EM only')
ax.errorbar(x=range(n_param), y=dk_nt_mu, yerr=dk_nt_er, label='neither')
ax.errorbar(x=range(n_param), y=dk_bo_mu, yerr=dk_bo_er, label='both')
if np.sum(wm_only_cond_p2) > 0:
    ax.errorbar(x=range(n_param), y=dk_wm_mu, yerr=dk_wm_er, label='WM only')
ax.set_title(f'Uncertainty, {cond_name}')
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('P(DK)')
ax.legend(fancybox=True)
f.tight_layout()
sns.despine()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-dk-by-dk.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

'''ma ~ correct in the EM only case'''

tma_crt_mu, tma_crt_er = np.zeros(n_param,), np.zeros(n_param,)
tma_incrt_mu, tma_incrt_er = np.zeros(n_param,), np.zeros(n_param,)
for t in range(n_param):
    tma_ = targ_act_cond_p2[em_only_cond_p2[:, t], t+pad_len_test]
    crt_ = corrects_cond_p2[em_only_cond_p2[:, t], t]
    tma_crt_mu[t], tma_crt_er[t] = compute_stats(tma_[crt_])
    tma_incrt_mu[t], tma_incrt_er[t] = compute_stats(tma_[~crt_])

f, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.errorbar(x=range(n_param), y=tma_crt_mu, yerr=tma_crt_er, label='correct')
ax.errorbar(x=range(n_param), y=tma_incrt_mu,
            yerr=tma_incrt_er, label='incorrect')
ax.set_ylim([0, None])
ax.legend()
ax.set_title(f'Target memory activation, {cond_name}')
ax.set_ylabel('Activation')
ax.set_xlabel('Time, recall phase')
sns.despine()
f.tight_layout()

'''ma ~ kdk in the EM only case'''

tma_k_mu, tma_k_er = np.zeros(n_param,), np.zeros(n_param,)
tma_dk_mu, tma_dk_er = np.zeros(n_param,), np.zeros(n_param,)
for t in range(n_param):
    tma_ = targ_act_cond_p2[em_only_cond_p2[:, t], t+pad_len_test]
    dk_ = dk_cond_p2[em_only_cond_p2[:, t], t]
    tma_k_mu[t], tma_k_er[t] = compute_stats(tma_[~dk_])
    tma_dk_mu[t], tma_dk_er[t] = compute_stats(tma_[dk_])

f, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.errorbar(x=range(n_param), y=tma_k_mu, yerr=tma_k_er, label='know')
ax.errorbar(x=range(n_param), y=tma_dk_mu, yerr=tma_dk_er, label='don\'t know')
ax.set_ylim([0, None])
ax.legend()
ax.set_title(f'Target memory activation, {cond_name}')
ax.set_ylabel('Activation')
ax.set_xlabel('Time, recall phase')
sns.despine()
f.tight_layout()

'''ma ~ mistake or not, when q in neither'''
# plt.plot(np.sum(mistakes_cond_p2, axis=0))
#
# # np.sum(np.logical_and(mistakes_cond_p2, em_only_cond_p2), axis=0)
# # np.sum(np.logical_and(mistakes_cond_p2, wm_only_cond_p2), axis=0)
# # np.sum(np.logical_and(mistakes_cond_p2, neither_cond_p2), axis=0)
# for t in range(n_param):
#     print(targ_act_cond_p2[neither_cond_p2[:, t], t+pad_len_test])
#     mistakes_cond_p2
# # np.sum(np.logical_and(mistakes_cond_p2, both_cond_p2), axis=0)


'''analyze the EM-only condition'''

f, ax = plt.subplots(1, 1, figsize=(6, 3.5))
plot_pred_acc_rcl(
    pa_em_mu, pa_em_er, pa_em_mu+dk_em_mu, p,
    f, ax,
    title=f'EM-based prediction performance, {cond_name}',
    baseline_on=False, legend_on=True,
)
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('Accuracy')
f.tight_layout()
sns.despine()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-pa-em-only.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


f, ax = plt.subplots(1, 1, figsize=(6, 3.5))
plot_pred_acc_rcl(
    pa_bo_mu, pa_bo_er, pa_bo_mu+dk_bo_mu, p,
    f, ax,
    title=f'WM+EM based prediction performance, {cond_name}',
    baseline_on=False, legend_on=True,
)
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('Accuracy')
ax.set_ylim([-0.05, 1.05])
f.tight_layout()
sns.despine()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-pa-both.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''signal detection analysis'''

# plot the max-score distribution for one time step
ms_lure = np.max(sim_lca_dict['NM']['lure'], axis=-1)
ms_targ = np.max(sim_lca_dict['DM']['targ'], axis=-1)
leg_ = ['NM', 'DM']

t = 2
bins = 50
dt_ = [ms_lure[:, T_part+t], ms_targ[:, T_part+t]]

f, ax = plt.subplots(1, 1, figsize=(6, 3))
for j, m_type in enumerate(memory_types):
    sns.distplot(
        dt_[j],
        # hist=False,
        bins=bins,
        # kde=False,
        # kde_kws={"shade": True},
        ax=ax, color=gr_pal[::-1][j]
    )
ax.legend(leg_, frameon=False,)
ax.set_title('Max score distribution')
ax.set_xlabel('Recall strength')
ax.set_ylabel('Counts')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
sns.despine()
fig_path = os.path.join(fig_dir, f'ms-dist-t{t}.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''signal detection, max score'''
# roc analysis
ms_lure = ms_lure[:np.shape(ms_targ)[0], :]
tprs, fprs, auc = compute_auc_over_time(ms_lure.T, ms_targ.T)

b_pal = sns.color_palette('Blues', n_colors=T_part)
f, axes = plt.subplots(2, 1, figsize=(5, 7))
for t in np.arange(T_part, T_total):
    axes[0].plot(fprs[t], tprs[t], color=b_pal[t-T_part])
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].set_title('ROC curves over time')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
axes[1].plot(auc[T_part:], color='black')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('AUC')
axes[1].set_title('AUC over time')
for ax in axes:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
f.tight_layout()
sns.despine()
fig_path = os.path.join(fig_dir, f'roc.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''compute inter-event similarity'''
ambiguity = np.zeros((n_examples, n_mem-1))
for i in np.arange(n_mem-1, n_examples):
    cur_mem_ids = np.arange(i-n_mem+1, i)
    prev_events = [targets[j] for j in cur_mem_ids]
    cur_event = targets[i]
    for j in range(n_mem-1):
        ambiguity[i, j] = compute_event_similarity(cur_event, prev_events[j])

# plot event similarity distribution
confusion_mu = np.mean(ambiguity, axis=1)
f, ax = plt.subplots(1, 1, figsize=(5, 3))
sns.distplot(confusion_mu, kde=False, ax=ax)
ax.set_xlabel('Similarity')
ax.set_ylabel('P')
sns.despine()
f.tight_layout()


'''performance metrics ~ ambiguity'''
corrects_by_cond, mistakes_by_cond, dks_by_cond = {}, {}, {}
corrects_by_cond_mu, mistakes_by_cond_mu, dks_by_cond_mu = {}, {}, {}
confusion_by_cond_mu = {}
for cond_name, cond_ids_ in cond_ids.items():
    # print(cond_name, cond_ids_)
    # collect the regressor by condiiton
    confusion_by_cond_mu[cond_name] = confusion_mu[cond_ids_]
    # collect the performance metrics
    corrects_by_cond[cond_name] = corrects[cond_ids_, :]
    mistakes_by_cond[cond_name] = mistakes[cond_ids_, :]
    dks_by_cond[cond_name] = dks[cond_ids_, :]
    # compute average for the recall phase
    corrects_by_cond_mu[cond_name] = np.mean(
        corrects_by_cond[cond_name][:, T_part:], axis=1)
    mistakes_by_cond_mu[cond_name] = np.mean(
        mistakes_by_cond[cond_name][:, T_part:], axis=1)
    dks_by_cond_mu[cond_name] = np.mean(
        dks_by_cond[cond_name][:, T_part:], axis=1)

# show regression model
# predictor: inter-event similarity
ind_var = confusion_by_cond_mu
# # set dependent var
# dep_var = corrects_by_cond_mu
# dep_var = mistakes_by_cond_mu
# dep_var = dks_by_cond_mu
# f, axes = plt.subplots(3, 1, figsize=(5, 11), sharex=True)
# for i, cond_name in enumerate(cond_ids.keys()):
#     print(i, cond_name)
#     sns.regplot(
#         ind_var[cond_name], dep_var[cond_name],
#         ax=axes[i]
#     )
#     axes[i].set_title(cond_name)
# ylims_ = get_ylim_bonds(axes)
# for ax in axes:
#     ax.set_ylim(ylims_)
#     ax.set_xlabel('Similarity')
# sns.despine()
# f.tight_layout()

''''''
dep_vars = {
    'Corrects': corrects_by_cond_mu, 'Errors': mistakes_by_cond_mu,
    'Uncertain': dks_by_cond_mu
}
cond_name = 'DM'

f, axes = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
for i, info in enumerate(dep_vars.keys()):
    print(i, cond_name)
    sns.regplot(
        ind_var[cond_name], dep_vars[info][cond_name],
        ax=axes[i]
    )
    axes[i].set_ylabel(info)
axes[0].set_title(cond_name)

ylims_ = get_ylim_bonds(axes)
for ax in axes:
    ax.set_ylim(ylims_)
    ax.set_xlabel('Similarity')
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'ambiguity-{cond_name}.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''weights'''
for name, wts in agent.named_parameters():
    wts_np = to_sqnp(wts)
    wts_norm = np.linalg.norm(wts_np)
    wts_mean = np.mean(wts_np)
    wts_min, wts_max = np.min(wts_np), np.max(wts_np)
    print(name, np.shape(wts_np))
    print(wts_norm, wts_mean, wts_min, wts_max)


'''t-RDM: raw similarity'''
data = C
trsm = {}
for cond_name in cond_ids.keys():
    if np.sum(cond_ids[cond_name]) == 0:
        continue
    else:
        data_cond_ = data[cond_ids[cond_name], :, :]
        trsm[cond_name] = compute_trsm(data_cond_)

f, axes = plt.subplots(3, 1, figsize=(7, 11))
for i, cond_name in enumerate(TZ_COND_DICT.values()):
    sns.heatmap(
        trsm[cond_name], cmap='viridis', square=True,
        ax=axes[i]
    )
    axes[i].axvline(T_part, color='red', linestyle='--')
    axes[i].axhline(T_part, color='red', linestyle='--')
    axes[i].set_title(f'TR-TR correlation, {cond_name}')
f.tight_layout()

'''pca the deicison activity'''

n_pcs = 5
data = DA
cond_name = 'DM'


# fit PCA
pca = PCA(n_pcs)
# np.shape(data)
# np.shape(data_cond)
data_cond = data[cond_ids[cond_name], :, :]
data_cond = data_cond[:, ts_predict, :]
targets_cond = targets[cond_ids[cond_name]]
mistakes_cond = mistakes_by_cond[cond_name]
dks_cond = dks[cond_ids[cond_name], :]

# Loop over timepoints
pca_cum_var_exp = np.zeros((np.sum(ts_predict), n_pcs))
for t in range(np.sum(ts_predict)):
    data_pca = pca.fit_transform(data_cond[:, t, :])
    pca_cum_var_exp[t] = np.cumsum(pca.explained_variance_ratio_)

    f, ax = plt.subplots(1, 1, figsize=(7, 5))
    # plot the data
    for y_val in range(p.y_dim):
        y_sel_op = y_val == targets_cond
        sel_op_ = np.logical_and(~dks[cond_ids[cond_name], t], y_sel_op[:, t])
        ax.scatter(
            data_pca[sel_op_, 0], data_pca[sel_op_, 1],
            marker='o', alpha=alpha,
        )
    ax.scatter(
        data_pca[dks[cond_ids[cond_name], t], 0],
        data_pca[dks[cond_ids[cond_name], t], 1],
        marker='o', color='grey', alpha=alpha,
    )
    ax.scatter(
        data_pca[mistakes_cond[:, t], 0], data_pca[mistakes_cond[:, t], 1],
        facecolors='none', edgecolors='red',
    )
    # add legend
    ax.legend(
        [f'choice {k}' for k in range(task.y_dim)] + ['uncertain', 'error'],
        fancybox=True, bbox_to_anchor=(1, .5), loc='center left'
    )
    # mark the plot
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'Pre-decision activity, time = {t}')
    sns.despine(offset=10)
    f.tight_layout()
    # fig_path = os.path.join(fig_dir, f'pca/da-t-{t}.png')
    # f.savefig(fig_path, dpi=150, bbox_to_anchor='tight')


# plot cumulative variance explained curve
t = -1
pc_id = 1
f, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(pca_cum_var_exp[t])
ax.set_title('First %d PCs capture %d%% of variance' %
             (pc_id+1, pca_cum_var_exp[t, pc_id]*100))
ax.axvline(pc_id, color='grey', linestyle='--')
ax.axhline(pca_cum_var_exp[t, pc_id], color='grey', linestyle='--')
ax.set_ylim([None, 1.05])
ytickval_ = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ytickval_])
ax.set_xticks(np.arange(n_pcs))
ax.set_xticklabels(np.arange(n_pcs)+1)
ax.set_ylabel('cum. var. exp.')
ax.set_xlabel('Number of components')
sns.despine(offset=5)
f.tight_layout()

# sns.heatmap(pca_cum_var_exp, cmap='viridis')
