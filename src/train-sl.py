import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.LCALSTM_v9_2 import LCALSTM as Agent
# from models import LCALSTM as Agent
from task import SequenceLearning
from exp_tz import run_tz
from analysis import compute_behav_metrics, compute_acc, compute_dk
from utils.io import build_log_path, save_ckpt, save_all_params, load_ckpt
from utils.utils import to_sqnp
from utils.params import P
from utils.constants import TZ_COND_DICT
from plt_helper import plot_pred_acc_full
# from sklearn.decomposition.pca import PCA
plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

'''learning to tz with a2c. e.g. cmd:
python -u train-tz.py --exp_name testing --subj_id 0 \
--penalty 4 --n_param 6 --n_hidden 64 --eta .1\
--n_epoch 300 --sup_epoch 50 --train_init_state 0 \
--log_root ../log/
'''

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='test', type=str)
parser.add_argument('--subj_id', default=99, type=int)
parser.add_argument('--n_param', default=6, type=int)
parser.add_argument('--n_branch', default=3, type=int)
parser.add_argument('--pad_len', default=0, type=int)
parser.add_argument('--penalty', default=4, type=int)
parser.add_argument('--p_rm_ob_enc', default=0, type=float)
parser.add_argument('--p_rm_ob_rcl', default=0, type=float)
parser.add_argument('--n_hidden', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--eta', default=0.1, type=float)
parser.add_argument('--n_mem', default=3, type=int)
parser.add_argument('--sup_epoch', default=100, type=int)
parser.add_argument('--n_epoch', default=300, type=int)
parser.add_argument('--n_examples', default=256, type=int)
parser.add_argument('--log_root', default='../log/', type=str)
args = parser.parse_args()
print(args)

# process args
exp_name = args.exp_name
subj_id = args.subj_id
n_param = args.n_param
n_branch = args.n_branch
pad_len = args.pad_len
penalty = args.penalty
p_rm_ob_enc = args.p_rm_ob_enc
p_rm_ob_rcl = args.p_rm_ob_rcl
n_hidden = args.n_hidden
learning_rate = args.lr
eta = args.eta
n_mem = args.n_mem
n_examples = args.n_examples
n_epoch = args.n_epoch
supervised_epoch = args.sup_epoch
log_root = args.log_root

# log_root = '../log/'
# exp_name = 'always-recall'
# subj_id = 1
# penalty = 2
# supervised_epoch = 100
# n_epoch = 300
# n_examples = 256
# n_param = 6
# n_branch = 3
# pad_len = 3
# n_hidden = 64
# learning_rate = 1e-3
# eta = .1
# p_rm_ob_enc = 2/n_param
# p_rm_ob_rcl = 2/n_param
# n_mem = 2


'''init'''
np.random.seed(subj_id)
torch.manual_seed(subj_id)

p = P(
    exp_name=exp_name, sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    penalty=penalty,
    p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
    n_hidden=n_hidden, lr=learning_rate, eta=eta, n_mem=n_mem
)
# init env
task = SequenceLearning(
    n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=p.env.pad_len,
    p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
)
# init agent
input_dim = task.x_dim
# input_dim = task.x_dim+2
agent = Agent(
    input_dim, p.net.n_hidden, p.a_dim, dict_len=p.net.n_mem
)
optimizer = torch.optim.Adam(
    agent.parameters(), lr=p.net.lr, weight_decay=0
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True
)

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)
# save experiment params initial weights
save_all_params(log_subpath['data'], p)
save_ckpt(0, log_subpath['ckpts'], agent, optimizer)

# load model
# epoch_load = None
# epoch_load = 300
# if epoch_load is not None:
#     agent, optimizer = load_ckpt(
#         epoch_load, log_subpath['ckpts'], agent, optimizer)
#     epoch_id = epoch_load-1
# else:
#     epoch_id = 0

'''task definition'''
log_freq = 20
Log_loss_critic = np.zeros(n_epoch,)
Log_loss_actor = np.zeros(n_epoch,)
Log_loss_sup = np.zeros(n_epoch,)
Log_return = np.zeros(n_epoch,)
Log_pi_ent = np.zeros(n_epoch,)
Log_acc = np.zeros((n_epoch, task.n_parts))
Log_mis = np.zeros((n_epoch, task.n_parts))
Log_dk = np.zeros((n_epoch, task.n_parts))
Log_cond = np.zeros((n_epoch, n_examples))

# epoch_id, i, t = 0, 0, 0
epoch_id = 0
for epoch_id in np.arange(epoch_id, n_epoch):
    time0 = time.time()
    # training objective
    supervised = epoch_id < supervised_epoch
    [results, metrics] = run_tz(
        agent, optimizer, task, p, n_examples,
        supervised=supervised, cond=None, learning=True, get_cache=False,
    )
    [dist_a, targ_a, _, Log_cond[epoch_id]] = results
    [Log_loss_sup[epoch_id], Log_loss_actor[epoch_id], Log_loss_critic[epoch_id],
     Log_return[epoch_id], Log_pi_ent[epoch_id]] = metrics
    # compute stats
    bm_ = compute_behav_metrics(targ_a, dist_a, task)
    Log_acc[epoch_id], Log_mis[epoch_id], Log_dk[epoch_id] = bm_
    acc_mu_pts_str = " ".join('%.2f' % i for i in Log_acc[epoch_id])
    dk_mu_pts_str = " ".join('%.2f' % i for i in Log_dk[epoch_id])
    mis_mu_pts_str = " ".join('%.2f' % i for i in Log_mis[epoch_id])
    # print
    runtime = time.time() - time0
    msg = '%3d | R: %.2f, acc: %s, dk: %s, mis: %s, ent: %.2f | ' % (
        epoch_id, Log_return[epoch_id],
        acc_mu_pts_str, dk_mu_pts_str, mis_mu_pts_str, Log_pi_ent[epoch_id])
    msg += 'L: a: %.2f c: %.2f, s: %.2f | t: %.2fs' % (
        Log_loss_actor[epoch_id], Log_loss_critic[epoch_id],
        Log_loss_sup[epoch_id], runtime)
    print(msg)
    # update lr scheduler
    if not supervised:
        neg_pol_score = np.mean(Log_mis[epoch_id]) - np.mean(Log_acc[epoch_id])
        scheduler.step(neg_pol_score)
    # save weights
    if np.mod(epoch_id+1, log_freq) == 0:
        save_ckpt(epoch_id+1, log_subpath['ckpts'], agent, optimizer)

'''plot learning curves'''
f, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
axes[0, 0].plot(Log_return)
axes[0, 0].set_ylabel('return')
axes[0, 0].axhline(0, color='grey', linestyle='--')
axes[0, 0].set_title(Log_return[-1])

axes[0, 1].plot(Log_pi_ent)
axes[0, 1].set_ylabel('entropy')

axes[1, 0].plot(Log_loss_actor, label='actor')
axes[1, 0].plot(Log_loss_critic, label='critic')
axes[1, 0].axhline(0, color='grey', linestyle='--')
axes[1, 0].legend()
axes[1, 0].set_ylabel('loss, rl')

axes[1, 1].plot(Log_loss_sup)
axes[1, 1].set_ylabel('loss, sup')

for ip in range(2):
    axes[2, ip].set_title(f'part {ip+1}')
    axes[2, ip].plot(Log_acc[:, ip], label='acc')
    axes[2, ip].plot(Log_acc[:, ip]+Log_dk[:, ip], label='acc+dk')
    axes[2, ip].plot(
        Log_acc[:, ip]+Log_dk[:, ip] + Log_mis[:, ip],
        label='acc+dk_err', linestyle='--', color='red'
    )
axes[2, -1].legend()
axes[2, 0].set_ylabel('% behavior')

for i, ax in enumerate(f.axes):
    ax.axvline(supervised_epoch, color='grey', linestyle='--')

axes[-1, 0].set_xlabel('Epoch')
axes[-1, 1].set_xlabel('Epoch')
sns.despine()
f.tight_layout()
fig_path = os.path.join(log_subpath['figs'], 'tz-lc.png')
f.suptitle('learning curves', fontsize=15)
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

'''plot performance'''
# prep data
cond_ids = {}
for cond_name_ in list(TZ_COND_DICT.values()):
    cond_id_ = TZ_COND_DICT.inverse[cond_name_]
    cond_ids[cond_name_] = Log_cond[-1, :] == cond_id_
    targ_a_ = targ_a[cond_ids[cond_name_], :]
    dist_a_ = dist_a[cond_ids[cond_name_], :]
    # compute performance for this condition
    acc_mu, acc_er = compute_acc(targ_a_, dist_a_, return_er=True)
    dk_mu = compute_dk(dist_a_)
    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    plot_pred_acc_full(
        acc_mu, acc_er, acc_mu+dk_mu,
        [n_param], p,
        f, ax,
        title=f'Performance on the TZ task: {cond_name_}',
    )
    fig_path = os.path.join(log_subpath['figs'], f'tz-acc-{cond_name_}.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
