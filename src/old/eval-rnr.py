import numpy as np
import os.path as osp
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import comb
from scipy.stats import sem, entropy
from models import LCALSTM
from utils.params import P
from data.ExpRun import process_cache
from data.StimGen import RNR_COND_DICT as cond_dict
from utils.io import build_log_path, load_ckpt, pickle_load_dict
from utils.constants import rnr_log_fnames
from analysis.utils import process_trial_type_info, \
    compute_correct_rate_wrapper, form_df, compute_mb, compute_rt, \
    extrat_trials, compute_baseline, compute_evidence, sigmoid
# from utils.utils import to_np
# from models import pick_action, get_reward, compute_returns
from analysis import compute_predacc, compute_dks, get_baseline
from matplotlib.ticker import FormatStrFormatter
sns.set(style='white', context='talk', palette='colorblind')

# from utils.utils import to_np

log_root = '../log/'

'''input args'''
subj_id = 0
penalty = 4
epoch_load = 100
n_epoch = 2000
n_mvs_rnr = 3
exp_name = 'multi-lures'

'''fixed params'''
# event parameters
n_param = 6
n_branch = 3
n_hidden = 64
learning_rate = 1e-3
# log params
p = P(
    exp_name=exp_name,
    n_param=n_param, n_branch=n_branch,
    penalty=penalty, n_hidden=n_hidden, lr=learning_rate,
)
p.env.rnr.n_mvs = n_mvs_rnr
n_mems = n_mvs_rnr-1

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)
ckpt_template, save_data_fname = rnr_log_fnames(epoch_load, n_epoch)
data_dict = pickle_load_dict(osp.join(log_subpath['rnr-data'], save_data_fname))

# unpack data
[C, V, Scalar, Vector] = data_dict['model_acts']
[Y_encs, Y_rcl, Y_hat] = data_dict['data_labels']
[memory_ids, trial_type_info] = data_dict['data_metadata']

'''analyze data'''

# compute trial type info
tt_info = process_trial_type_info(trial_type_info)
trial_types, n_trial_types, trial_ids_set, n_trials = tt_info
n_trials_total = np.sum(n_trials)
trial_type_names = list(cond_dict.values())
print('number of trials per condition: ', n_trials)

# split the data by trial type
memory_ids_atp = [memory_ids[trial_ids_] for trial_ids_ in trial_ids_set]
C_atp = [extrat_trials(C, trial_ids_) for trial_ids_ in trial_ids_set]
V_atp = [extrat_trials(V, trial_ids_) for trial_ids_ in trial_ids_set]
Scalar_atp = np.array([extrat_trials(Scalar, trial_ids_)
                       for trial_ids_ in trial_ids_set])
Vector_atp = np.array([extrat_trials(Vector, trial_ids_)
                       for trial_ids_ in trial_ids_set])
Y_rcl_atp = [extrat_trials(Y_rcl, trial_ids_)
             for trial_ids_ in trial_ids_set]
Y_encs = [[Y_encs[i][m].data.numpy() for m in range(n_mems)]
          for i in range(n_trials_total)]
Y_encs_atp = np.squeeze([extrat_trials(Y_encs, trial_ids_)
                         for trial_ids_ in trial_ids_set])

np.shape(Y_hat)
np.shape(Y_rcl)
# compute correct rate
corrects = [
    compute_correct_rate_wrapper(
        Y_rcl, Y_hat[:, :, :-1], trial_ids_set[j])[:, :-1]
    for j in range(n_trial_types)
]
# form df for plotting
corrects_dfs = [form_df(corrects_j) for corrects_j in corrects]

# compute lstm smilarity between cell states vs. memories
model_sim = [None] * n_trial_types
for j in range(n_trial_types):
    model_sim[j] = compute_evidence(
        C_atp[j], V_atp[j],
        Scalar_atp[j][:, :, 0], Scalar_atp[j][:, :, 1], Scalar_atp[j][:, :, 2],
        p.net.recall_func, p.net.kernel
    )

#

n_chunks = p.env.event_len//p.net.enc_size
n_mems_total = n_chunks * n_mems

# compute stats , NR trials
model_sim_nr = np.mean(model_sim[1], axis=1)
# compute, R trials
model_sim_r = model_sim[0]
model_sim_r_targ = np.zeros((p.env.event_len, n_chunks, n_trials[0]))
model_sim_r_lure = np.zeros((p.env.event_len, (n_mems-1)*n_chunks, n_trials[0]))
# separate target vs. lures
for i in range(n_trials[0]):
    targ_id_ = int(memory_ids_atp[0][i])
    targ_ids = np.arange(targ_id_*n_chunks, (targ_id_+1)*n_chunks)
    lure_ids = list(set(range(n_mems_total)).difference(set(targ_ids)))
    model_sim_r_targ[:, :, i] = model_sim_r[:, targ_ids, i]
    model_sim_r_lure[:, :, i] = model_sim_r[:, lure_ids, i]


'''plot params'''
dpi = 150
ci_val = 99
n_se = 3
alpha_val = .05
# target-lure pallete
tl_pals = sns.color_palette(
    'colorblind', n_colors=4, desat=1)[-n_trial_types:]
# trial type params
tt_pals = sns.color_palette(n_colors=n_trial_types)

# behavioral chance
predacc_chance = 1/n_branch
predacc_baseline = get_baseline(n_param, predacc_chance)
# predacc_baseline = np.append(predacc_baseline, 1)

'''compute model similarity / recall strength
'''

# compute stats
# NR
model_sim_nr_mu = np.mean(model_sim_nr, axis=1)

model_sim_nr_eb = sem(model_sim_nr, axis=1) * n_se
# R
model_sim_r_lure_ = np.mean(model_sim_r_lure, axis=1)
model_sim_r_targ_ = np.mean(model_sim_r_targ, axis=1)

model_sim_r_targ_mu = np.mean(model_sim_r_targ_, axis=1)
model_sim_r_targ_eb = sem(model_sim_r_targ_, axis=1) * n_se
model_sim_r_lure_mu = np.mean(model_sim_r_lure_, axis=1)
model_sim_r_lure_eb = sem(model_sim_r_lure_, axis=1) * n_se

f, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.errorbar(
    range(p.env.event_len), y=model_sim_r_targ_mu, yerr=model_sim_r_targ_eb,
    color=tl_pals[0])
ax.errorbar(
    range(p.env.event_len), y=model_sim_r_lure_mu, yerr=model_sim_r_lure_eb,
    color=tl_pals[1])
ax.errorbar(
    range(p.env.event_len), y=model_sim_nr_mu, yerr=model_sim_nr_eb,
    linestyle='--', color=tl_pals[1])
f.legend(['recall, target', 'recall, lures', 'no recall, lures'],
         frameon=False, bbox_to_anchor=(.95, .85))
ax.axhline(0, color='grey', linestyle='--')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.set_title('memory activation during recall for target vs. lures')
ax.set_ylabel('Recall strength')
ax.set_xlabel('Time')
ax.set_xticks(np.arange(0, p.env.event_len, 5))
sns.despine()
f.tight_layout()

# img_name = f'e{epoch_load}-rmdnd{lesion_recall}-recall-str.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)

'''scratch'''
np.shape(corrects)
np.shape(model_sim_nr)
np.shape(model_sim_r_targ)
np.shape(model_sim_r_lure)

'''visualize scalar signals'''


def plot_scalar_signal(Scalar_j_, color, ylabel_, ax):
    Scalar_j_mu = np.mean(Scalar_j_[:, :-1], axis=0)
    Scalar_j_se = sem(Scalar_j_[:, :-1], axis=0) * n_se
    ax.errorbar(range(p.env.event_len-1), Scalar_j_mu, Scalar_j_se, color=color)
    # ax.set_title(ylabel_)
    ax.set_ylabel(ylabel_)
    # ax.set_xlabel('Time')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    f.tight_layout()
    sns.despine(offset=10)


n_ssig = 3
ssig_names = ['Input strength', 'Leak', 'Competition']
f, axes = plt.subplots(n_ssig, 1, figsize=(6, 2.5*n_ssig), sharex=True)
for j in range(n_trial_types):
    for i_s in range(n_ssig):
        plot_scalar_signal(
            Scalar_atp[j][:, :, i_s],
            tt_pals[j], ssig_names[i_s], axes[i_s]
        )
axes[0].set_title('Recall/LCA parameters')
axes[-1].set_xlabel('Time')
axes[0].legend(trial_type_names, frameon=False)
# img_name = f'e{epoch_load}-rmdnd{lesion_recall}-recall-params.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)


"""
visualize prediction accs results, all phases
"""

# plot
ci_val = 95
f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(predacc_baseline[1:], color='black', linestyle='--', alpha=.5)
for j in range(n_trial_types):
    sns.lineplot(
        x='time', y='val',
        ci=ci_val,
        data=corrects_dfs[j],
        ax=ax
    )
# ax.axvline(pad_length-1, color='red', linestyle='--', alpha=.3)
ax.axhline(predacc_chance, color='grey', linestyle='--', alpha=.5)
ax.axhline(1, color='grey', linestyle='--', alpha=.5)

title_text = f'Next state prediction accuracy'
ax.set_title(title_text)
ax.set_ylabel('Prediction accuracy')
ax.set_xlabel('Time')
f.legend(['baseline']+trial_type_names,
         frameon=False, bbox_to_anchor=(.95, .55))
ax.set_ylim([predacc_chance-.05, None])
# if pad_length > 1:
#     ax.set_xlim([pad_length-1, None])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xticks(np.arange(0, p.env.event_len, 5))
sns.despine()
f.tight_layout()
# img_name = f'e{epoch_load}-rmdnd{lesion_recall}-pacc-recall.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)

'''
entropy of the prediction
'''

ents = [np.zeros((n_trials[i], n_param))for i in range(n_trial_types)]

for j in range(n_trial_types):
    for tid in range(n_trials[j]):
        Y_probs_ji = sigmoid(Y_hat[trial_ids_set[j]][tid, :, :].T)
        ents[j][tid, :] = [
            entropy(Y_probs_ji[:, t]) for t in range(n_param)
        ]

f, ax = plt.subplots(1, 1, figsize=(7, 4))
for j in range(n_trial_types):
    ax.errorbar(
        x=range(n_param),
        y=np.mean(ents[j], axis=0),
        yerr=sem(ents[j], axis=0) * 3
    )
# for b in event_bounds_plt:
#     ax.axvline(b, color='red', alpha=.3, linestyle='--')
ax.set_ylabel(r'Entropy of $\hat{y}_t$')
ax.set_title('Uncertainty over time')
ax.set_xlabel('Time')
f.legend(trial_type_names, frameon=False, bbox_to_anchor=(1, .9))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

f.tight_layout()
sns.despine()

# img_name = f'e{epoch_load}-rmdnd{lesion_recall}-ent-all.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)

"""
visualize results, gate values
"""

# # list all gates
# names_plt = [
#     'flush / forget cell states',
#     'write to cell state',
#     'read from cell state'
# ]
#
# f, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
# for j in range(n_trial_types):
#     for i, ax in enumerate(axes):
#         data_ = Vector_atp[j, :, :-1, :, i]
#         mean_gate_val = np.mean(np.mean(data_, axis=-1), axis=0)
#         std_gate_val = np.mean(sem(data_, axis=-1)*n_se, axis=0)
#         ax.errorbar(x=range(len(mean_gate_val)),
#                     y=mean_gate_val, yerr=std_gate_val)
#         ax.set_title(names_plt[i])
#         ax.set_ylabel('Value')
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axes[0].legend(trial_type_names, frameon=False)
# axes[-1].set_xlabel('Time')
# f.tight_layout()
# sns.despine()
#
# img_name = f'e{epoch_load}-rmdnd{lesion_recall}-vsigs.png'
# f.savefig(osp.join(img_path, img_name))

# '''signal detction metric - max score, compare trial types
# '''
#
#
# # np.shape(model_sim)
# n_bins = 100
# histrange = (0, 1)
# max_acts = np.max(model_sim, axis=2)
# tprs, fprs, auc = compute_auc_over_time(
#     max_acts[1], max_acts[0], n_bins=n_bins, histrange=histrange)
#
# # compute the average/max AUC over time
# avg_auc = np.mean(auc)
# max_auc = np.max(auc)
#
# cur_pal = sns.color_palette('Blues', n_colors=n_param)
#
# f, axes = plt.subplots(2, 1, figsize=(5, 7))
# for t in range(n_param):
#     axes[0].plot(fprs[t], tprs[t], color=cur_pal[t])
# axes[0].set_xlabel('FPR')
# axes[0].set_ylabel('TPR')
# axes[0].set_title('ROC curves over time')
# axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
#
# axes[1].plot(auc, color='black')
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('AUC')
# axes[1].set_title('AUC over time')
# # axes[1].set_title('AUC over time, mean = %.2f' % (np.mean(auc)))
# axes[1].axhline(.5, linestyle='--', color='grey')
# axes[1].set_ylim([.4, 1.05])
#
# for ax in axes:
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
# f.tight_layout()
# sns.despine()
# img_name = f'e{epoch_load}-auc-ms.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
#
# n_bins = 200
# vmin = np.min(max_acts)
# vmax = np.max(max_acts)
# # vmin = np.min(max_acts[:, t, :])
# # vmax = np.max(max_acts[:, t, :])
# bw = (vmax - vmin) / n_bins
#
# for t in range(n_param):
#     f, ax = plt.subplots(1, 1, figsize=(6, 3))
#     for j in range(n_trial_types):
#         p = sns.kdeplot(
#             max_acts[j, t, :],
#             shade=True,
#             bw=bw,
#             ax=ax,
#             shade_lowest=True,
#         )
#     ax.set_title('Max score distributions')
#     ax.set_xlabel('Max score')
#     ax.set_ylabel('Counts')
#     # ax.set_xlim([-.2, 1.2])
#     # ax.set_xlim([vmin*.9, vmax*1.1])
#     # ax.set_ylim([0, 9])
#     f.legend(trial_type_names, frameon=False, bbox_to_anchor=(1, .85))
#     sns.despine()
#     img_name = f'e{epoch_load}_ms-dist-t{t}.png'
#     f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
#
# #
# # '''cems 2019 - scratch'''
# #
# # f, axes = plt.subplots(1, 2, figsize=(8.5, 4))
# # axes[0].plot(auc, color='black')
# # axes[0].plot([2], auc[2], color='red', marker='o')
# # axes[0].set_xlabel('Time')
# # axes[0].set_ylabel('AUC')
# # axes[0].axhline(.5, linestyle='--', color='grey')
# # axes[0].set_ylim([.45, 1.05])
# # axes[0].text(-.1, 1, 'A', fontsize=28, fontweight='bold')
# # for j in range(n_trial_types):
# #     p = sns.kdeplot(
# #         max_acts[j, 3, :],
# #         shade=True,
# #         bw=bw,
# #         ax=axes[1],
# #         shade_lowest=True,
# #     )
# # axes[1].set_ylabel('Frequency')
# # axes[1].set_xlabel('Max score')
# # axes[1].set_ylim([0, 40])
# # axes[1].set_yticks([0, 35])
# # axes[1].set_yticklabels([0, .1])
# # axes[1].legend(trial_type_names, frameon=False, bbox_to_anchor=(1.1, 1))
# # axes[1].text(-.01, 36.5, 'B', fontsize=28, fontweight='bold')
# # sns.despine()
# # f.tight_layout()
# # f.savefig('temp/auc.png', bbox_inches='tight', dpi=dpi)
#
# '''signal detction metric - max score, compare target vs. lure in R trials
# '''
#
# n_bins = 100
# histrange = (0, 1)
#
# # max_acts = np.max(model_sim, axis=2)
# tprs, fprs, auc = compute_auc_over_time(
#     model_sim_r_lure, model_sim_r_targ,
#     n_bins=n_bins, histrange=histrange
# )
# # np.shape(model_sim_r_lure)
# # np.shape(model_sim_r_targ)
#
# cur_pal = sns.color_palette('Blues', n_colors=n_param)
#
# f, axes = plt.subplots(2, 1, figsize=(5, 7))
# for t in range(n_param):
#     axes[0].plot(fprs[t], tprs[t], color=cur_pal[t])
# axes[0].set_xlabel('FPR')
# axes[0].set_ylabel('TPR')
# axes[0].set_title('ROC curves over time')
# axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
#
# axes[1].plot(auc, color='black')
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('AUC')
# # axes[1].set_title('AUC over time, mean = %.2f' % (np.mean(auc)))
# axes[1].set_title('AUC over time')
# axes[1].axhline(.5, linestyle='--', color='grey')
# axes[1].set_ylim([.4, 1.05])
#
# for ax in axes:
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
# f.tight_layout()
# sns.despine()
#
# img_name = f'e{epoch_load}-r-tl-auc.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
#
# # np.shape(data_plt)
# d_pack = [model_sim_r_targ, model_sim_r_lure]
# np.shape(model_sim_r_targ)
# n_bins = 50
# # vmin = np.min(data_plt)
# # vmax = np.max(data_plt)
# vmin = np.min(max_acts[:, t, :])
# vmax = np.max(max_acts[:, t, :])
# bw = (vmax - vmin) / n_bins
#
# for t in range(n_param):
#     f, ax = plt.subplots(1, 1, figsize=(6, 3))
#     for j in range(n_trial_types):
#         p = sns.kdeplot(
#             d_pack[j][t, :],
#             shade=True,
#             bw=bw,
#             ax=ax,
#             color=tl_pals[j]
#         )
#         # p = sns.distplot(
#         #     max_acts[j, t, :],
#         #     kde=False,
#         #     bins=n_bins,
#         #     # shade=True,
#         #     # bw=bw,
#         #     ax=ax
#         # )
#     ax.set_title('Target lure distributions, recall trials')
#     ax.set_xlabel('Recall strength')
#     ax.set_ylabel('Counts')
#     # ax.set_xlim([-.2, 1.2])
#     # ax.set_xlim([vmin*.9, vmax*1.1])
#     # ax.set_ylim([0, 190])
#     # f.legend(CONDITIONS, frameon=False)
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     # f.tight_layout()
#     sns.despine()
#     img_name = f'e{epoch_load}_r-tl-dist-t{t}.png'
#     f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
#
# # print(log_path)
# #
# # j = 0
# # np.shape(Y_rcl_atp[j])
#
# # for r trials, compute the similarity between target vs. lure
# j = 0
# # compute the number of params shared across lures
# n_p_shared = np.zeros((n_trials[j], n_lures))
# # compute the memory benefit for all trials
# memory_benefits = np.zeros((n_trials[j], ))
#
# for i in range(n_trials[j]):
#     # get the target info
#     targ_id = memory_ids_atp[j][i]
#     targ_mem_i = Y_encs_atp[j][i][targ_id]
#     targ_mem_vec_i = np.sum(targ_mem_i, axis=0)
#     # loop over all lures, compute target-lure similarity
#     lure_mem_vecs_i = []
#     for m in set(range(n_mems_rnr)).difference({targ_id}):
#         lure_mem_vec_m = np.sum(Y_encs_atp[j][i][m], axis=0)
#         lure_mem_vecs_i.append(lure_mem_vec_m)
#     n_p_shared[i, :] = [
#         np.sum(np.logical_and(lure_mem_vec_m, targ_mem_vec_i))
#         for lure_mem_vec_m in lure_mem_vecs_i
#     ]
#     # compute mb
#     memory_benefits[i] = compute_mb(corrects[j][i], predacc_chance)
#
# n_p_shared_mu = np.mean(n_p_shared, axis=1)
# n_p_shared_mu_uniques = np.unique(n_p_shared_mu)
# n_mu = len(n_p_shared_mu_uniques)
#
# n_p_shared_max = np.max(n_p_shared, axis=1)
# n_p_shared_max_uniques = np.unique(n_p_shared_max)
# n_max = len(n_p_shared_max_uniques)
#
# # split memory benefit
#
# splitter = n_p_shared_max
# splitter_vals = n_p_shared_max_uniques
# n = n_max
#
# # splitter = n_p_shared_mu
# # splitter_vals = n_p_shared_mu_uniques
# # n = n_mu
#
# mb_mu_by_sp = [
#     np.mean(memory_benefits[splitter == k])
#     for k in splitter_vals
# ]
# mb_se_by_sp = [
#     sem(memory_benefits[splitter == k]) * n_se
#     for k in splitter_vals
# ]
#
# # compute RT
# rt_thres = 3
# rts = np.zeros((n_trial_types, n_trials[j]))
# for j in range(n_trial_types):
#     for i in range(n_trials[j]):
#         rts[j, i] = compute_rt(corrects[j][i], rt_thres, p.env.event_len)
# rts_mu_by_sp = [
#     [np.mean(rts[j, splitter == k]) for k in splitter_vals]
#     for j in range(n_trial_types)
# ]
# rts_se_by_sp = [
#     [sem(rts[j, splitter == k]) * n_se for k in splitter_vals]
#     for j in range(n_trial_types)
# ]
#
# # f, ax = plt.subplots(1, 1, figsize=(5, 4))
# # ax.bar(x=range(n), height=mb_mu_by_sp, yerr=mb_se_by_sp)
# # ax.set_xticks(range(n))
# # ax.set_xticklabels([int(z) for z in n_p_shared_max_uniques])
# # ax.set_title('Performance ~ ambiguity')
# # ax.set_xlabel('Target-lure similarity')
# # ax.set_ylabel('Memory benefit')
# # f.tight_layout()
# # sns.despine()
#
# # sns.jointplot(x=n_p_shared_mu, y=memory_benefits, kind="kde")
# # [r, p] = pearsonr(n_p_shared_mu, memory_benefits)
# # sns.jointplot(x=n_p_shared_max, y=memory_benefits, kind="kde")
#
# # sns.distplot(n_p_shared_mu[memory_benefits< 0])
#
# f, ax = plt.subplots(1, 1, figsize=(5, 4))
# [r, p] = pearsonr(n_p_shared_mu, memory_benefits)
# x_jit = np.random.normal(size=(n_trials[j]), scale=.0)
# y_jit = np.random.normal(size=(n_trials[j]), scale=.0)
# ax.scatter(
#     x=n_p_shared_mu+x_jit,
#     y=memory_benefits+y_jit,
#     marker='.', alpha=.2
# )
# sns.regplot(
#     x=n_p_shared_mu, y=memory_benefits,
#     scatter=False, order=1, ci=99,
# )
# ax.axhline(1, color='grey', linestyle='--', alpha=.3)
# ax.axhline(0, color='grey', linestyle='--', alpha=.3)
# title_text = f"""Performance ~ ambiguity
# r = %.2f, p = %.5f
# """ % (r, p)
# # ax.set_title(title_text)
# ax.set_xlabel('Target-lure similarity')
# ax.set_ylabel('Memory benefit')
# f.tight_layout()
# sns.despine(offset=10)
# img_name = f'e{epoch_load}_mb-amb.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
#
# # data = {"mb": memory_benefits, "p": n_p_shared_mu}
# # model = smf.ols(formula='a ~ np.power(b, 2)', data=data).fit()
#
# f, ax = plt.subplots(1, 1, figsize=(4, 3))
# sns.distplot(
#     rts[0],
#     bins=int(np.max(rts)-np.min(rts)),
#     norm_hist=True,
#     kde=False,
#     # kde=True, kde_kws={'bw': .6},
#     ax=ax
# )
# # ax.set_title('Recall time distribution')
# ax.set_ylabel('Frequency')
# ax.set_xlabel('Recall time')
# ax.set_xlim([0, None])
# f.tight_layout()
# sns.despine()
# img_name = f'e{epoch_load}_rtdist_thres-{rt_thres}.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
#
# f, ax = plt.subplots(1, 1, figsize=(5, 4))
# ax.bar(
#     x=range(n), height=rts_mu_by_sp[j],
#     # yerr=rts_se_by_sp[j]
# )
# ax.set_xticks(range(n))
# ax.set_ylim([0, None])
# ax.set_xticklabels([int(z) for z in n_p_shared_max_uniques])
# # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.set_title('Recall time ~ ambiguity')
# ax.set_xlabel('max(target-lure similarity)')
# ax.set_ylabel('Recall time')
# f.tight_layout()
# sns.despine()
# img_name = f'e{epoch_load}_rt-amb.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
#
# """compute target-lure similarity, split by amb level"""
#
# amb_bound = np.median(n_p_shared_max_uniques)
# amb_high_ = n_p_shared_max_uniques[n_p_shared_max_uniques > amb_bound]
# amb_high = np.array([i in amb_high_ for i in n_p_shared_max])
# amb_low = np.logical_not(amb_high)
#
# # '''signal detction metric - max score, compare target vs. lure in R trials
# # '''
# #
# # np.shape(model_sim_r_lure)
# #
# # amb_conds = [amb_high, amb_low]
# # amb_conds_str = ['high', 'low']
# # jj = 0
# #
# # n_bins = 100
# # histrange = (0, 1)
# #
# # # for jj in range(len(amb_conds)):
# # d_pack = [model_sim_r_targ[:, amb_conds[jj]],
# #           model_sim_r_lure[:, amb_conds[jj]]]
# #
# # tprs, fprs, auc = compute_auc_over_time(
# #     d_pack[1], d_pack[0], n_bins=n_bins, histrange=histrange
# # )
# # cur_pal = sns.color_palette('Blues', n_colors=n_param)
# #
# # f, axes = plt.subplots(2, 1, figsize=(5, 7))
# # for t in range(n_param):
# #     axes[0].plot(fprs[t], tprs[t], color=cur_pal[t])
# # axes[0].set_xlabel('FPR')
# # axes[0].set_ylabel('TPR')
# # axes[0].set_title(f'ROC curves over time, {amb_conds_str[jj]} ambiguity')
# # axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
# #
# # axes[1].plot(auc, color='black')
# # axes[1].set_xlabel('Time')
# # axes[1].set_ylabel('AUC')
# # # axes[1].set_title('AUC over time, mean = %.2f' % (np.mean(auc)))
# # axes[1].set_title(f'AUC over time, {amb_conds_str[jj]} ambiguity')
# # axes[1].axhline(.5, linestyle='--', color='grey')
# # axes[1].set_ylim([.4, 1.05])
# #
# # for ax in axes:
# #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# # axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
# # f.tight_layout()
# # sns.despine()
# #
# # img_name = f'e{epoch_load}-r-tl-auc-amblv-{amb_conds_str[jj]}.png'
# # f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
# #
# #
# # amb_bound = 3
# # amb_high_ = n_p_shared_max_uniques[n_p_shared_max_uniques > amb_bound]
# # amb_high = np.array([i in amb_high_ for i in n_p_shared_max])
# # amb_low = np.logical_not(amb_high)
# # amb_conds_str = ['high', 'low']
# # amb_conds = [amb_high, amb_low]
# # jj = 1
# # d_pack = [model_sim_r_targ[:, amb_conds[jj]],
# #           model_sim_r_lure[:, amb_conds[jj]]]
# #
# #
# # n_bins = 50
# #
# # vmin = np.min(max_acts[:, t, :])
# # vmax = np.max(max_acts[:, t, :])
# # bw = (vmax - vmin) / n_bins
# # t = 2
# # # for t in range(n_param):
# # f, ax = plt.subplots(1, 1, figsize=(6, 3))
# # for j in range(n_trial_types):
# #     p = sns.kdeplot(
# #         d_pack[j][t, :],
# #         shade=True,
# #         bw=bw,
# #         ax=ax,
# #         color=tl_pals[j]
# #     )
# #     # p = sns.distplot(
# #     #     d_pack[j][t, :],
# #     #     kde=True,
# #     #     bins=n_bins,
# #     #     ax=ax
# #     # )
# # ax.set_title(
# #     f'Target lure distributions, recall trials, {amb_conds_str[jj]} ambiguity')
# # ax.set_xlabel('Recall strength')
# # ax.set_ylabel('Counts')
# # # ax.set_xlim([-.2, 1.2])
# # # ax.set_xlim([vmin*.9, vmax*1.1])
# # # ax.set_ylim([0, 190])
# # # f.legend(CONDITIONS, frameon=False)
# # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# # # f.tight_layout()
# # sns.despine()
# # # img_name = f'e{epoch_load}_r-tl-dist-t{t}-amb-{amb_conds_str[jj]}.png'
# # # f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)
