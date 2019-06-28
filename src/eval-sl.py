import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models import LCALSTM as Agent
from task import SequenceLearning
from utils.params import P
from utils.utils import to_sqnp, to_np, to_sqpth, to_pth
from utils.io import build_log_path, load_ckpt

# from models.DND import compute_similarities, transform_similarities
from exp_tz import run_tz
from sklearn.decomposition.pca import PCA
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_trsm, compute_cell_memory_similarity, create_sim_dict, \
    compute_auc_over_time, compute_event_similarity
from plt_helper import plot_tz_pred_acc, get_ylim_bonds
from matplotlib.ticker import FormatStrFormatter
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

log_root = '../log/'
exp_name = 'ohctx-longerseq-emnokey'
# exp_name = 'ohctx-branch5'

subj_id = 0
penalty = 1
supervised_epoch = 300
epoch_load = 600
# n_epoch = 500
# n_examples = 256
n_param = 12
n_branch = 3
n_hidden = 128
learning_rate = 1e-3
eta = .1
p_rm_ob_enc = .2
p_rm_ob_rcl = .2
n_mem = 5

np.random.seed(subj_id)
torch.manual_seed(subj_id)

'''init'''
p = P(
    exp_name=exp_name, sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch,
    penalty=penalty,
    p_rm_ob_enc=p_rm_ob_enc,
    p_rm_ob_rcl=p_rm_ob_rcl,
    n_hidden=n_hidden, lr=learning_rate, eta=eta, n_mem=n_mem
)
# init env
task = SequenceLearning(
    p.env.n_param, p.env.n_branch,
    n_rm_fixed=False,
    p_rm_ob_enc=p_rm_ob_enc,
    p_rm_ob_rcl=p_rm_ob_rcl,
)
# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)

# load the agent back
agent = Agent(task.x_dim, p.net.n_hidden, p.a_dim, dict_len=p.net.n_mem)
# load model
agent, optimizer = load_ckpt(epoch_load, log_subpath['ckpts'], agent)


'''eval'''
# training objective
n_examples = 512
[results, metrics] = run_tz(
    agent, optimizer, task, p, n_examples,
    supervised=False, learning=False
)
[log_dist_a, Y, log_cache, log_cond] = results


'''predefine/compute some constants'''
# precompute some constants
event_bonds = [p.env.tz.event_ends[0]+1]
n_conds = len(p.env.tz.cond_dict)
memory_types = ['targ', 'lure']
# plot
alpha = .5
n_se = 3
# colors
gr_pal = sns.color_palette('colorblind')[2:4]
# sns.palplot(gr_pal)

'''upack results'''
# skip examples untill em buffer is full
non_nm_trials = np.where(log_cond != p.env.tz.cond_dict.inverse['NM'])[0]
n_examples_skip = non_nm_trials[n_mem+1]
# update the number of examples
n_examples = n_examples - n_examples_skip

# n_exampls
log_dist_a = log_dist_a[n_examples_skip:, :, :]
Y = Y[n_examples_skip:, :, :]
log_cond = log_cond[n_examples_skip:]
# compute trial ids
cond_ids = {}
for cond_name_ in list(p.env.tz.cond_dict.values()):
    cond_id_ = p.env.tz.cond_dict.inverse[cond_name_]
    cond_ids[cond_name_] = log_cond == cond_id_

# convert data to numpy
inpt = torch.zeros((n_examples, task.T_total))
leak = torch.zeros((n_examples, task.T_total))
comp = torch.zeros((n_examples, task.T_total))
C = np.zeros((n_examples, task.T_total, p.net.n_hidden))
H = np.zeros((n_examples, task.T_total, p.net.n_hidden))
M = np.zeros((n_examples, task.T_total, p.net.n_hidden))
CM = np.zeros((n_examples, task.T_total, p.net.n_hidden))
DA = np.zeros((n_examples, task.T_total, p.net.n_hidden))
V = [None] * n_examples


for i, ii in enumerate(np.arange(n_examples_skip, n_examples+n_examples_skip)):
    for t in range(task.T_total):
        [vector_signal, scalar_signal, misc] = log_cache[ii][t]
        [inpt[i, t], leak[i, t], comp[i, t]] = scalar_signal
        [h_t, m_t, cm_t, des_act_t, V_i] = misc
        H[i, t, :] = to_sqnp(h_t)
        M[i, t, :] = to_sqnp(m_t)
        CM[i, t, :] = to_sqnp(cm_t)
        DA[i, t, :] = to_sqnp(des_act_t)
        V[i] = V_i

C = CM - M
inpt = to_sqnp(inpt)
leak = to_sqnp(leak)
comp = to_sqnp(comp)
Y_np = to_np(Y)

# pre-compute some stuff
actions = np.argmax(log_dist_a, axis=-1)
targets = np.argmax(to_sqnp(Y), axis=-1)
#
dks = actions == p.dk_id
corrects = targets == actions
mistakes = np.logical_and(targets != actions, ~dks)

'''plot behavioral performance'''

for cond_name_ in list(p.env.tz.cond_dict.values()):
    Y_ = Y_np[cond_ids[cond_name_], :]
    log_dist_a_ = log_dist_a[cond_ids[cond_name_], :]
    # compute performance for this condition
    acc_mu, acc_er = compute_acc(Y_, log_dist_a_, return_er=True)
    dk_mu = compute_dk(log_dist_a_)
    # plot
    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    plot_tz_pred_acc(
        acc_mu, acc_er, acc_mu+dk_mu,
        [p.env.tz.event_ends[0]+1], p,
        f, ax,
        title=f'Performance on the TZ task: {cond_name_}',
    )
    fig_path = os.path.join(log_subpath['figs'], f'tz-acc-{cond_name_}.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

'''compare LCA params across conditions'''


def plot_time_course_for_all_conds(
        matrix, ax,
        axis1_start=0, xlabel=None, ylabel=None, title=None,
        frameon=False, add_legend=True,
):
    for i, cond_name in enumerate(p.env.tz.cond_dict.values()):
        submatrix_ = matrix[cond_ids[cond_name], axis1_start:]
        M_, T_ = np.shape(submatrix_)
        mu_, er_ = compute_stats(submatrix_, axis=0, n_se=n_se)
        ax.errorbar(x=range(T_), y=mu_, yerr=er_, label=cond_name)
    if add_legend:
        ax.legend(frameon=frameon)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


f, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
plot_time_course_for_all_conds(
    inpt, axes[0], axis1_start=task.T_part,
    title='"need" for episodic memories', ylabel='input strength'
)
plot_time_course_for_all_conds(
    leak, axes[1], axis1_start=task.T_part,
    title='leakiness of the memories', ylabel='leak'
)
plot_time_course_for_all_conds(
    comp, axes[2], axis1_start=task.T_part,
    title='competition across memories', ylabel='competition'
)
axes[-1].set_xlabel('Time, recall phase')
axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
sns.despine()
f.tight_layout()
fig_path = os.path.join(log_subpath['figs'], f'tz-lca-param.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


'''compute cell memory similarity evidence '''
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

# plot
# sim_plt_ = sim_cos_stats
sim_plt_ = sim_lca_stats
f, axes = plt.subplots(3, 1, figsize=(5, 8))
for i, c_name in enumerate(cond_ids.keys()):
    for m_type in memory_types:
        if m_type == 'targ' and c_name == 'NM':
            continue
        color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
        axes[i].errorbar(
            x=range(task.T_part),
            y=sim_plt_[c_name][m_type]['mu'][task.T_part:],
            yerr=sim_plt_[c_name][m_type]['er'][task.T_part:],
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
f.tight_layout()
sns.despine()
fig_path = os.path.join(log_subpath['figs'], f'tz-memact.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

'''check the MS distribution'''
ms_lure = np.max(sim_lca_dict['NM']['lure'], axis=-1)
ms_targ = np.max(sim_lca_dict['DM']['targ'], axis=-1)

t = 6
bins = 10
dt_ = [ms_lure[:, task.T_part+t], ms_targ[:, task.T_part+t]]

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
ax.set_title('max score distributions')
ax.set_xlabel('Recall strength')
ax.set_ylabel('Counts')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
sns.despine()
# img_name = f'e{epoch_load}_r-tl-dist-t{t}.png'
# f.savefig(osp.join(img_path, img_name), bbox_inches='tight', dpi=dpi)


'''signal detection, max score'''
ms_targ = ms_targ[:np.shape(ms_lure)[0], :]
tprs, fprs, auc = compute_auc_over_time(ms_lure.T, ms_targ.T)

b_pal = sns.color_palette('Blues', n_colors=task.T_part)
f, axes = plt.subplots(2, 1, figsize=(5, 7))
for t in np.arange(task.T_part, task.T_total):
    axes[0].plot(fprs[t], tprs[t], color=b_pal[t-task.T_part])
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].set_title('ROC curves over time')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
axes[1].plot(auc[task.T_part:], color='black')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('AUC')
axes[1].set_title('AUC over time')
for ax in axes:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
f.tight_layout()
sns.despine()


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
        corrects_by_cond[cond_name][:, task.T_part:], axis=1)
    mistakes_by_cond_mu[cond_name] = np.mean(
        mistakes_by_cond[cond_name][:, task.T_part:], axis=1)
    dks_by_cond_mu[cond_name] = np.mean(
        dks_by_cond[cond_name][:, task.T_part:], axis=1)

covariate = corrects_by_cond_mu
covariate = mistakes_by_cond_mu
covariate = dks_by_cond_mu
f, axes = plt.subplots(3, 1, figsize=(5, 10))
for i, cond_name in enumerate(cond_ids.keys()):
    print(i, cond_name)
    sns.regplot(
        confusion_by_cond_mu[cond_name],
        covariate[cond_name],
        ax=axes[i]
    )
    axes[i].set_title(cond_name)
ylims_ = get_ylim_bonds(axes)
for ax in axes:
    ax.set_ylim(ylims_)
    ax.set_xlabel('Similarity')
sns.despine()
f.tight_layout()


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
for i, cond_name in enumerate(p.env.tz.cond_dict.values()):
    sns.heatmap(
        trsm[cond_name], cmap='viridis', square=True,
        ax=axes[i]
    )
    axes[i].axvline(event_bonds[0], color='red', linestyle='--')
    axes[i].axhline(event_bonds[0], color='red', linestyle='--')
    axes[i].set_title(f'TR-TR similarity, {cond_name}')
f.tight_layout()

'''pca the deicison activity'''

n_pcs = 5
t = 12
data = DA
cond_name = 'DM'

# fit PCA
pca = PCA(n_pcs)
data_cond = data[cond_ids[cond_name], :, :]
targets_cond = targets[cond_ids[cond_name]]
mistakes_ = mistakes_by_cond[cond_name]

# Loop over timepoints
pca_cum_var_exp = np.zeros((task.T_total, n_pcs))
for t in range(task.T_total):
    H_pca = pca.fit_transform(data_cond[:, t, :])
    pca_cum_var_exp[t] = np.cumsum(pca.explained_variance_ratio_)

    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    # plot the data
    for y_val in range(p.y_dim):
        y_sel_op = y_val == targets_cond
        sel_op_ = np.logical_and(~dks[cond_ids[cond_name], t], y_sel_op[:, t])
        ax.scatter(
            H_pca[sel_op_, 0], H_pca[sel_op_, 1],
            marker='o', alpha=alpha,
        )
    ax.scatter(
        H_pca[dks[cond_ids[cond_name], t], 0],
        H_pca[dks[cond_ids[cond_name], t], 1],
        marker='o', color='grey', alpha=alpha,
    )
    ax.scatter(
        H_pca[mistakes_[:, t], 0], H_pca[mistakes_[:, t], 1],
        facecolors='none', edgecolors='red',
    )
    # add legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        [f'choice {k}' for k in range(task.y_dim)] + ['don\'t know', 'mistake'],
        fancybox=True, bbox_to_anchor=(1, .5), loc='center left'
    )
    # mark the plot
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'Pre-decision activity, time = {t}')
    sns.despine(offset=10)
    f.tight_layout()

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
