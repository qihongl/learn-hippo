import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.LCALSTM_v9 import LCALSTM as Agent
# from models import LCALSTM as Agent
from scipy.stats import sem, pearsonr
from task import SequenceLearning
from exp_tz import run_tz
from utils.params import P
from utils.utils import to_sqnp, to_np, to_sqpth, to_pth
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, load_ckpt
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_trsm, compute_cell_memory_similarity, create_sim_dict, \
    compute_auc_over_time, compute_event_similarity, batch_compute_true_dk, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip,\
    compute_cell_memory_similarity_stats, sep_by_obj_uncertainty

from plt_helper import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds,\
    plot_time_course_for_all_conds, show_weight_stats
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

log_root = '../log/'
# exp_name = 'pred-delay'
exp_name = 'july9_v9'

subj_id = 0
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
# loading params
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = .3
pad_len_load = -1
# testing params
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
show_weight_stats(agent)

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

'''precompute some constants'''
# figure out max n-time-steps across for all trials
T_part = n_param + pad_len_test
T_total = T_part * task.n_parts
#
n_conds = len(TZ_COND_DICT)
memory_types = ['targ', 'lure']
ts_predict = np.array([t % T_part >= pad_len_test for t in range(T_total)])

'''skip examples untill EM is full'''
n_examples_skip = compute_n_trials_to_skip(log_cond_, p)
n_examples = n_examples_test - n_examples_skip
# skip
dist_a = dist_a_[n_examples_skip:]
Y = Y_[n_examples_skip:]
log_cond = log_cond_[n_examples_skip:]
log_cache = log_cache_[n_examples_skip:]
true_dk_wm = true_dk_wm_[n_examples_skip:]
true_dk_em = true_dk_em_[n_examples_skip:]

'''organize results to analyzable form'''
cond_ids = get_trial_cond_ids(log_cond)
activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
[C, H, M, CM, DA, V] = activity_
[inpt, leak, comp] = ctrl_param_

# onehot to int
actions = np.argmax(dist_a, axis=-1)
targets = np.argmax(Y, axis=-1)
# compute performance
dks = actions == p.dk_id
corrects = targets == actions
mistakes = np.logical_and(targets != actions, ~dks)


# %%
'''plotting params'''
alpha = .5
n_se = 3
# colors
gr_pal = sns.color_palette('colorblind')[2:4]
# make dir to save figs
fig_dir = os.path.join(log_subpath['figs'], f'delay-{pad_len_test}')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)


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


f, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
plot_time_course_for_all_conds(
    inpt, cond_ids, axes[0], axis1_start=T_part,
    title='"need" for episodic memories', ylabel='input strength'
)
plot_time_course_for_all_conds(
    leak, cond_ids, axes[1], axis1_start=T_part,
    title='leakiness of the memories', ylabel='leak'
)
plot_time_course_for_all_conds(
    comp, cond_ids, axes[2], axis1_start=T_part,
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
sim_cos_stats = compute_cell_memory_similarity_stats(sim_cos_dict, cond_ids)
sim_lca_stats = compute_cell_memory_similarity_stats(sim_lca_dict, cond_ids)


# plot target/lure activation for all conditions
sim_stats_plt = {'LCA': sim_lca_stats, 'cosine': sim_cos_stats}
for ker_name, sim_stats_plt_ in sim_stats_plt.items():
    # print(ker_name, sim_stats_plt_)
    f, axes = plt.subplots(3, 1, figsize=(5, 8))
    for i, c_name in enumerate(cond_ids.keys()):
        for m_type in memory_types:
            if m_type == 'targ' and c_name == 'NM':
                continue
            color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
            axes[i].errorbar(
                x=range(T_part),
                y=sim_stats_plt_[c_name][m_type]['mu'][T_part:],
                yerr=sim_stats_plt_[c_name][m_type]['er'][T_part:],
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
    fig_path = os.path.join(fig_dir, f'tz-memact-{ker_name}.png')
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

f, axes = plt.subplots(2, 1, figsize=(6, 7))

mu_, er_ = compute_stats(targ_act_cond_p2, n_se=3)
axes[0].plot(targ_act_cond_p2.T, alpha=.1, color=gr_pal[0])
axes[0].errorbar(x=range(n_param), y=mu_, yerr=er_, color='black')
axes[0].set_xlabel('Time, recall phase')
axes[0].set_ylabel('Memory activation')
axes[0].set_title(f'Target activation, {cond_name}')

sorted_targ_act_cond_p2 = np.sort(targ_act_cond_p2, axis=1)[:, ::-1]
mu_, er_ = compute_stats(sorted_targ_act_cond_p2, n_se=3)
axes[1].plot(sorted_targ_act_cond_p2.T, alpha=.1, color=gr_pal[0])
axes[1].errorbar(x=range(n_param), y=mu_, yerr=er_, color='black')
axes[1].set_ylabel('Memory activation')
axes[1].set_title(f'Sorted')

sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-targact-lca.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

f, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(targ_act_cond_p2[:5, :].T)
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('Memory activation')
ax.set_title(f'Target activation, {cond_name}')
sns.despine()
f.tight_layout()

# recall peak distribution
recall_peak_times = np.argmax(targ_act_cond_p2, axis=1)
f, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.set_title(f'Target activation peak time, {cond_name}')
sns.violinplot(
    recall_peak_times,
    color=gr_pal[0], ax=ax)
ax.set_xlabel('Time')
ax.set_ylabel('Freq')
sns.despine()
f.tight_layout()


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
obj_uncertainty_info = [em_only_cond_p2, wm_only_cond_p2, neither_cond_p2,
                        both_cond_p2]

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


# use objective uncertainty metric to decompose LCA params in the EM condition
inpt_cond_p2 = inpt[cond_ids[cond_name], T_part:]
leak_cond_p2 = leak[cond_ids[cond_name], T_part:]
comp_cond_p2 = comp[cond_ids[cond_name], T_part:]
all_lca_param_cond_p2 = {
    'input strength': inpt_cond_p2,
    'leak': leak_cond_p2, 'competition': comp_cond_p2
}

f, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
for i, (param_name, param_cond_p2) in enumerate(all_lca_param_cond_p2.items()):
    print(i, param_name, param_cond_p2)
    param_cond_p2_stats = sep_by_obj_uncertainty(
        param_cond_p2[pad_len_test:], obj_uncertainty_info, n_se=n_se)
    for key, [mu_, er_] in param_cond_p2_stats.items():
        if not np.all(np.isnan(mu_)):
            axes[i].errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
    axes[i].set_ylabel(f'{param_name}')
    axes[i].legend(fancybox=True)

axes[-1].set_xlabel('Time, recall phase')
axes[0].set_title(f'LCA params, {cond_name}')
f.tight_layout()
sns.despine()


# use CURRENT uncertainty to predict memory activation

targ_act_cond_p2_stats = sep_by_obj_uncertainty(
    targ_act_cond_p2[pad_len_test:], obj_uncertainty_info, n_se=n_se)

f, ax = plt.subplots(1, 1, figsize=(8, 5))
for key, [mu_, er_] in targ_act_cond_p2_stats.items():
    if not np.all(np.isnan(mu_)):
        ax.errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
ax.set_ylabel(f'{param_name}')
ax.legend(fancybox=True)
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
acc_cond_p2_stats = sep_by_obj_uncertainty(
    corrects_cond_p2, obj_uncertainty_info, n_se=n_se)

f, ax = plt.subplots(1, 1, figsize=(6, 4))
for key, [mu_, er_] in acc_cond_p2_stats.items():
    if not np.all(np.isnan(mu_)):
        ax.errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
ax.set_title(f'Performance, {cond_name}')
ax.set_xlabel('Time, recall phase')
ax.set_ylabel('Accuracy')
ax.legend(fancybox=True)
f.tight_layout()
sns.despine()
fig_path = os.path.join(fig_dir, f'tz-{cond_name}-pa-by-dk.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''dk w.r.t prediction source (EM, WM)'''

dk_cond_p2_stats = sep_by_obj_uncertainty(
    dk_cond_p2, obj_uncertainty_info, n_se=n_se)

f, ax = plt.subplots(1, 1, figsize=(6, 4))
for key, [mu_, er_] in dk_cond_p2_stats.items():
    if not np.all(np.isnan(mu_)):
        ax.errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
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
    acc_cond_p2_stats['EM only'][0], acc_cond_p2_stats['EM only'][1],
    acc_cond_p2_stats['EM only'][0]+dk_cond_p2_stats['EM only'][0],
    p, f, ax,
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
    acc_cond_p2_stats['both'][0], acc_cond_p2_stats['both'][1],
    acc_cond_p2_stats['both'][0]+dk_cond_p2_stats['both'][0],
    p, f, ax,
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

t = t_recall_peak
bins = 30
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
dep_vars = {
    'Corrects': corrects_by_cond_mu, 'Errors': mistakes_by_cond_mu,
    'Uncertain': dks_by_cond_mu
}
c_pal = sns.color_palette(n_colors=3)
f, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True)
for col_id, cond_name in enumerate(cond_ids.keys()):
    for row_id, info_name in enumerate(dep_vars.keys()):
        sns.regplot(
            ind_var[cond_name], dep_vars[info_name][cond_name],
            scatter_kws={'alpha': .5, 'marker': '.', 's': 15},
            x_jitter=.025, y_jitter=.05,
            color=c_pal[col_id],
            ax=axes[row_id, col_id]
        )
        corr, pval = pearsonr(
            ind_var[cond_name], dep_vars[info_name][cond_name]
        )
        str_ = 'r = %.2f, p = %.2f' % (corr, pval)
        str_ = str_+'*' if pval < .05 else str_
        str_ = cond_name + '\n' + str_ if row_id == 0 else str_
        axes[row_id, col_id].set_title(str_)
        axes[row_id, 0].set_ylabel(info_name)
        axes[row_id, col_id].set_ylim([-.05, 1.05])

    axes[-1, col_id].set_xlabel('Similarity')
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'ambiguity-by-cond.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


# '''ambiguity regression, sep by objective uncertainty'''
# cond_name = 'DM'
# confusion_cond = confusion_by_cond_mu[cond_name]
# confusion_cond_ext = np.tile(confusion_cond, (15, 1)).T
# f, ax = plt.subplots(1, 1, figsize=(5, 4))
# sns.regplot(
#     dk_cond_p2[em_only_cond_p2], confusion_cond_ext[em_only_cond_p2],
#     scatter_kws={'alpha': .5, 'marker': '.', 's': 15},
#     x_jitter=.025, y_jitter=.05,
#     # color=c_pal[col_id],
#     ax=ax
# )
# np.mean(dk_cond_p2, axis=1)
# np.shape(mistakes_cond_p2)


'''t-RDM: raw similarity'''
data = C
trsm = {}
for cond_name in cond_ids.keys():
    if np.sum(cond_ids[cond_name]) == 0:
        continue
    else:
        data_cond_ = data[cond_ids[cond_name], :, :]
        trsm[cond_name] = compute_trsm(data_cond_)

f, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
for i, cond_name in enumerate(TZ_COND_DICT.values()):
    sns.heatmap(
        trsm[cond_name], cmap='viridis', square=True,
        xticklabels=5, yticklabels=5,
        ax=axes[i]
    )
    axes[i].axvline(T_part, color='red', linestyle='--')
    axes[i].axhline(T_part, color='red', linestyle='--')
    axes[i].set_title(f'TR-TR correlation, {cond_name}')
    axes[i].set_ylabel('Time')
axes[-1].set_xlabel('Time')
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
