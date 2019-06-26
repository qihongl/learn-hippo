import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from models import LCALSTM as Agent
from task import SequenceLearning
from analysis import compute_acc, compute_dk, entropy, compute_stats, compute_trsm
from utils.params import P
from utils.utils import to_sqnp
from utils.io import build_log_path, load_ckpt
from plt_helper import plot_tz_pred_acc
from exp_tz import run_tz
from sklearn.decomposition.pca import PCA
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

'''learning to tz with a2c. e.g. cmd:
python -u train-rnr.py --exp_name testing --subj_id 0 \
--penalty 4 --n_param 6 --n_hidden 64 --eta .1\
--n_epoch 300 --sup_epoch 50 --train_init_state 0 \
--log_root ../log/
'''

# parser = argparse.ArgumentParser()
# parser.add_argument('--exp_name', default='test', type=str)
# parser.add_argument('--subj_id', default=99, type=int)
# parser.add_argument('--penalty', default=4, type=int)
# parser.add_argument('--p_rm_ob_enc', default=0, type=float)
# parser.add_argument('--p_rm_ob_rcl', default=0, type=float)
# parser.add_argument('--n_param', default=6, type=int)
# parser.add_argument('--n_branch', default=2, type=int)
# parser.add_argument('--n_hidden', default=64, type=int)
# parser.add_argument('--lr', default=1e-3, type=float)
# parser.add_argument('--eta', default=0.1, type=float)
# parser.add_argument('--sup_epoch', default=100, type=int)
# parser.add_argument('--n_epoch', default=300, type=int)
# parser.add_argument('--n_examples', default=256, type=int)
# parser.add_argument('--log_root', default='../log/', type=str)
# args = parser.parse_args()
# print(args)
#
# # process args
# exp_name = args.exp_name
# subj_id = args.subj_id
# penalty = args.penalty
# p_rm_ob_enc = args.p_rm_ob_enc
# p_rm_ob_rcl = args.p_rm_ob_rcl
# n_param = args.n_param
# n_branch = args.n_branch
# n_hidden = args.n_hidden
# learning_rate = args.lr
# eta = args.eta
# n_examples = args.n_examples
# n_epoch = args.n_epoch
# supervised_epoch = args.sup_epoch
# log_root = args.log_root

log_root = '../log/'
exp_name = 'embedrnrtz'
subj_id = 0
penalty = 0
supervised_epoch = 300
epoch_load = 700
# n_epoch = 500
# n_examples = 256
n_param = 10
n_branch = 3
n_hidden = 128
learning_rate = 1e-3
eta = .1
p_rm_ob_enc = 2/n_param
p_rm_ob_rcl = 2/n_param

np.random.seed(subj_id)
torch.manual_seed(subj_id)

'''init'''
p = P(
    exp_name=exp_name, sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch,
    penalty=penalty,
    p_rm_ob_enc=p_rm_ob_enc,
    p_rm_ob_rcl=p_rm_ob_rcl,
    n_hidden=n_hidden, lr=learning_rate, eta=eta
)
# init env
context_dim = n_param + n_branch
# context_dim = 10
task = SequenceLearning(
    p.env.n_param, p.env.n_branch,
    context_onehot=True,
    append_context=True,
    context_dim=context_dim,
    n_rm_fixed=False,
    p_rm_ob_enc=p_rm_ob_enc,
    p_rm_ob_rcl=p_rm_ob_rcl,
)
# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)

# load the agent back
agent = Agent(task.x_dim, p.net.n_hidden, p.a_dim)
# load model
optimizer = torch.optim.Adam(agent.parameters(), lr=p.net.lr)
agent, optimizer = load_ckpt(epoch_load, log_subpath['ckpts'], agent, optimizer)
# agent.eval()
# reinit the agent
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
#     verbose=True
# )


'''eval'''
# training objective
n_examples = 256
[results, metrics] = run_tz(
    agent, optimizer, task, p, n_examples,
    supervised=False, learning=True
)
[log_dist_a, Y, log_cache, log_cond] = results
# [Log_loss_sup[epoch_id], Log_loss_actor[epoch_id], Log_loss_critic[epoch_id],
#  Log_return[epoch_id], Log_pi_ent[epoch_id]] = metrics

# # log message
# # compute stats
# bm_ = compute_behav_metrics(Y, log_dist_a, p)
# Log_acc[epoch_id], Log_mis[epoch_id], Log_dk[epoch_id] = bm_
# acc_mu_pts_str = " ".join('%.2f' % i for i in Log_acc[epoch_id])
# dk_mu_pts_str = " ".join('%.2f' % i for i in Log_dk[epoch_id])
# mis_mu_pts_str = " ".join('%.2f' % i for i in Log_mis[epoch_id])
# # print
# runtime = time.time() - time0
# msg = '%3d | R: %.2f, acc: %s, dk: %s, mis: %s, ent: %.2f | ' % (
#     epoch_id, Log_return[epoch_id],
#     acc_mu_pts_str, dk_mu_pts_str, mis_mu_pts_str, Log_pi_ent[epoch_id])
# msg += 'L: a: %.2f c: %.2f, s: %.2f | t: %.2fs' % (
#     Log_loss_actor[epoch_id], Log_loss_critic[epoch_id],
#     Log_loss_sup[epoch_id], runtime)
# print(msg)

'''upack results'''
cond_ids = {}
for cond_name_ in list(p.env.tz.cond_dict.values()):
    cond_id_ = p.env.tz.cond_dict.inverse[cond_name_]
    cond_ids[cond_name_] = log_cond == cond_id_

inpt = torch.zeros((n_examples, task.T_total))
leak = torch.zeros((n_examples, task.T_total))
comp = torch.zeros((n_examples, task.T_total))
C = np.zeros((n_examples, task.T_total, p.net.n_hidden))
H = np.zeros((n_examples, task.T_total, p.net.n_hidden))
M = np.zeros((n_examples, task.T_total, p.net.n_hidden))
CM = np.zeros((n_examples, task.T_total, p.net.n_hidden))
DA = np.zeros((n_examples, task.T_total, p.net.n_hidden))

for i in range(n_examples):
    for t in range(task.T_total):
        [vector_signal, scalar_signal, misc] = log_cache[i][t]
        [inpt[i, t], leak[i, t], comp[i, t]] = scalar_signal
        [h_t, m_t, cm_t, des_act_t, V] = misc
        H[i, t, :] = to_sqnp(h_t)
        M[i, t, :] = to_sqnp(m_t)
        CM[i, t, :] = to_sqnp(cm_t)
        DA[i, t, :] = to_sqnp(des_act_t)

C = CM - M
inpt = to_sqnp(inpt)
leak = to_sqnp(leak)
comp = to_sqnp(comp)
# precompute some constants
event_bonds = [p.env.tz.event_ends[0]+1]

'''plot behavioral performance'''


for cond_name_ in list(p.env.tz.cond_dict.values()):
    Y_ = to_sqnp(Y)[cond_ids[cond_name_], :]
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
    # fig_path = os.path.join(log_subpath['figs'], f'tz-acc-{cond_name_}.png')
    # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

'''plot lca params'''
# cond_name = 'RM'

n_se = 3
f, axes = plt.subplots(3, 1, figsize=(7, 11))
for i, ax in enumerate(axes):
    cond_name = p.env.tz.cond_dict[i]
    inpw_mu, inpw_er = compute_stats(
        inpt[cond_ids[cond_name], task.T_part:], axis=0, n_se=n_se)
    leak_mu, leak_er = compute_stats(
        leak[cond_ids[cond_name], task.T_part:], axis=0, n_se=n_se)
    comp_mu, comp_er = compute_stats(
        comp[cond_ids[cond_name], task.T_part:], axis=0, n_se=n_se)
    ax.errorbar(x=range(task.T_part), y=inpw_mu, yerr=inpw_er, label='input w')
    ax.errorbar(x=range(task.T_part), y=leak_mu, yerr=leak_er, label='leak')
    ax.errorbar(x=range(task.T_part), y=comp_mu,
                yerr=comp_er, label='competition')
    # ax.axvline(event_bonds, color='red', linestyle='--')
    ax.set_title(f'{cond_name}')
    ax.legend()
    ax.set_ylabel('Magnitude')

axes[-1].set_xlabel('Time, recall phase')
sns.despine()
f.tight_layout()

'''t-RDM'''

data = C
trsm = {}
for cond_name in cond_ids.keys():
    if np.sum(cond_ids[cond_name]) == 0:
        continue
    else:
        data_cond_ = data[cond_ids[cond_name], :, :]
        trsm[cond_name] = compute_trsm(data_cond_)

cond_name = 'DM'
f, ax = plt.subplots(1, 1)
sns.heatmap(
    trsm[cond_name], cmap='viridis', square=True,
    # center=0,
    ax=ax
)
ax.axvline(event_bonds[0], color='red', linestyle='--')
ax.axhline(event_bonds[0], color='red', linestyle='--')
ax.set_title(f'TR-TR, similarity, {cond_name}')


'''memory - cell state correlation'''


def cosine_similarity(u, v):
    return u @ v / np.linalg.norm(u) / np.linalg.norm(v)


h_0, _ = agent.get_init_states()
h_0 = to_sqnp(h_0)
h_0 = np.random.normal(size=(agent.hidden_dim,)) * .1

corr_b = np.zeros(task.T_total,)
corr_m = np.zeros(task.T_total,)


data = C
cond_name = 'NM'
data_cond_ = data[cond_ids[cond_name], :, :]

for i in range(len(data_cond_)):
    H_i = data_cond_[i, :, :]
    mem = H_i[task.T_part-1, ]
    for t in range(task.T_total):
        corr_b[t] += cosine_similarity(h_0, H_i[t, :])
        # np.corrcoef(h_0, H_i[t, :])[0, 1]
        corr_m[t] += cosine_similarity(mem, H_i[t, :])
        # np.corrcoef(mem, H_i[t, :])[0, 1]
corr_b /= len(data_cond_)
corr_m /= len(data_cond_)

f, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.plot(corr_m)
ax.plot(corr_b)
ax.axhline(0, color='grey', linestyle='--')
ax.axvline(event_bonds[0], color='red', linestyle='--')
ax.set_title(f'{cond_name}')
sns.despine()


# f, ax = plt.subplots(1, 1, figsize=(6, 3))
# ax.plot(trsm[cond_id_][5, :])
# ax.axhline(0, color='grey', linestyle='--')
# ax.axvline(event_bonds[0]-1, color='red', linestyle='--')
# ax.set_xlabel('Time')
# ax.set_ylabel('Correlation')
# ax.set_title('similarity: cell state vs. memory, ' +
#              f'{p.env.tz.cond_dict[cond_id_]}')
# sns.despine()
#
#
# np.shape(log_dist_a)
# np.shape(H)


actions = np.argmax(log_dist_a[:, :, :], axis=-1)
targets = np.argmax(to_sqnp(Y), axis=-1)
dks = actions == p.dk_id
np.shape(dks)

n_pcs = 5
pca = PCA(n_pcs)
data = DA
cond_name = 'RM'
data_cond = data[cond_ids[cond_name], :, :]
targets_cond = targets[cond_ids[cond_name]]

t = 5
alpha = .6

pca_cum_var_exp = np.zeros((task.T_total, n_pcs))
h_0 = to_sqnp(agent.get_init_states()[0]).reshape(1, -1)
for t in range(task.T_total):

    H_pca = pca.fit_transform(data_cond[:, t, :])
    pca_cum_var_exp[t] = np.cumsum(pca.explained_variance_ratio_)
    h_0_pca = pca.transform(h_0)

    f, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.axvline(0, linestyle='--', color='grey')
    ax.axhline(0, linestyle='--', color='grey')
    for y_val in range(p.y_dim):
        y_sel_op = y_val == targets_cond
        sel_op_ = np.logical_and(~dks[cond_ids[cond_name], t], y_sel_op[:, t])
        ax.scatter(
            H_pca[sel_op_, 0], H_pca[sel_op_, 1],
            marker='o', alpha=alpha,
        )
        # f.legend(['0', '1'])
    ax.scatter(
        H_pca[dks[cond_ids[cond_name], t], 0],
        H_pca[dks[cond_ids[cond_name], t], 1],
        marker='x', color='black', alpha=alpha,
    )

    # ax.scatter(
    #     h_0_pca[:, 0], h_0_pca[:, 1], color='red'
    # )

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'Pre-decision activity, time = {t}')
    sns.despine()


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
