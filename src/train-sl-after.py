import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models import LCALSTM_after as Agent
from task import SequenceLearning
from exp_tz import run_tz
from analysis import compute_behav_metrics, compute_acc, compute_dk
from vis import plot_pred_acc_full
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, save_ckpt, save_all_params,  \
    pickle_save_dict, get_test_data_dir, get_test_data_fname

plt.switch_backend('agg')
sns.set(style='white', palette='colorblind', context='talk')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='test', type=str)
parser.add_argument('--subj_id', default=99, type=int)
parser.add_argument('--n_param', default=16, type=int)
parser.add_argument('--n_branch', default=4, type=int)
parser.add_argument('--pad_len', default=-1, type=int)
parser.add_argument('--def_prob', default=None, type=float)
parser.add_argument('--n_def_tps', default=0, type=int)
parser.add_argument('--enc_size', default=16, type=int)
parser.add_argument('--dict_len', default=None, type=int)
parser.add_argument('--penalty', default=4, type=int)
parser.add_argument('--penalty_random', default=1, type=int)
parser.add_argument('--penalty_discrete', default=1, type=int)
parser.add_argument('--penalty_onehot', default=0, type=int)
parser.add_argument('--normalize_return', default=1, type=int)
parser.add_argument('--attach_cond', default=0, type=int)
parser.add_argument('--p_rm_ob_enc', default=0.3, type=float)
parser.add_argument('--p_rm_ob_rcl', default=0, type=float)
parser.add_argument('--similarity_max', default=.9, type=float)
parser.add_argument('--similarity_min', default=0, type=float)
parser.add_argument('--n_hidden', default=194, type=int)
parser.add_argument('--n_hidden_dec', default=128, type=int)
parser.add_argument('--lr', default=7e-4, type=float)
parser.add_argument('--eta', default=0.1, type=float)
parser.add_argument('--cmpt', default=0.4, type=float)
parser.add_argument('--n_event_remember', default=2, type=int)
parser.add_argument('--sup_epoch', default=1, type=int)
parser.add_argument('--n_epoch', default=2, type=int)
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
def_prob = args.def_prob
n_def_tps = args.n_def_tps
enc_size = args.enc_size
dict_len = args.dict_len
penalty = args.penalty
penalty_random = args.penalty_random
penalty_discrete = args.penalty_discrete
penalty_onehot = args.penalty_onehot
normalize_return = args.normalize_return
attach_cond = args.attach_cond
p_rm_ob_enc = args.p_rm_ob_enc
p_rm_ob_rcl = args.p_rm_ob_rcl
similarity_max = args.similarity_max
similarity_min = args.similarity_min
n_hidden = args.n_hidden
n_hidden_dec = args.n_hidden_dec
learning_rate = args.lr
cmpt = args.cmpt
eta = args.eta
n_event_remember = args.n_event_remember
n_examples = args.n_examples
n_epoch = args.n_epoch
supervised_epoch = args.sup_epoch
log_root = args.log_root


'''init'''
seed_val = subj_id
np.random.seed(seed_val)
torch.manual_seed(seed_val)

p = P(
    exp_name=exp_name, subj_id=subj_id,
    sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    def_prob=def_prob, n_def_tps=n_def_tps,
    enc_size=enc_size, dict_len=dict_len, n_event_remember=n_event_remember,
    penalty=penalty, penalty_random=penalty_random,
    penalty_discrete=penalty_discrete, penalty_onehot=penalty_onehot,
    normalize_return=normalize_return, attach_cond=attach_cond,
    p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
    n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
    lr=learning_rate, eta=eta, cmpt=cmpt,
)
# init env
task = SequenceLearning(
    n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=p.env.pad_len,
    p_rm_ob_enc=p.env.p_rm_ob_enc, p_rm_ob_rcl=p.env.p_rm_ob_rcl,
    def_path=p.env.def_path, def_prob=p.env.def_prob, def_tps=p.env.def_tps,
    similarity_cap_lag=p.n_event_remember,
    similarity_max=similarity_max, similarity_min=similarity_min,
)
# init agent
agent = Agent(
    input_dim=task.x_dim, output_dim=p.a_dim,
    rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
    dict_len=p.net.dict_len, cmpt=p.net.cmpt
)

optimizer_sup = torch.optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler_sup = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_sup, factor=1 / 2, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True)

optimizer_rl = torch.optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler_rl = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_rl, factor=1 / 2, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True)


# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)
# save experiment params initial weights
save_all_params(log_subpath['data'], p)
save_ckpt(0, log_subpath['ckpts'], agent, optimizer_sup)


'''task definition'''
log_freq = 200
Log_loss_critic = np.zeros(n_epoch,)
Log_loss_actor = np.zeros(n_epoch,)
Log_loss_sup = np.zeros(n_epoch,)
Log_return = np.zeros(n_epoch,)
Log_pi_ent = np.zeros(n_epoch,)
Log_acc = np.zeros((n_epoch, task.n_parts))
Log_mis = np.zeros((n_epoch, task.n_parts))
Log_dk = np.zeros((n_epoch, task.n_parts))
Log_cond = np.zeros((n_epoch, n_examples))

epoch_id = 0
for epoch_id in np.arange(epoch_id, n_epoch):
    time0 = time.time()
    # training objective
    supervised = epoch_id < supervised_epoch
    if supervised:
        optimizer = optimizer_sup
    else:
        optimizer = optimizer_rl

    [results, metrics] = run_tz(
        agent, optimizer, task, p, n_examples,
        supervised=supervised, fix_cond=None, learning=True, get_cache=False,
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
    if supervised:
        scheduler_sup.step(Log_loss_sup[epoch_id])
    else:
        neg_pol_score = np.mean(Log_mis[epoch_id]) - np.mean(Log_acc[epoch_id])
        scheduler_rl.step(neg_pol_score)

    # save weights
    if np.mod(epoch_id + 1, log_freq) == 0:
        save_ckpt(epoch_id + 1, log_subpath['ckpts'], agent, optimizer)


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
    avg_err = np.mean(Log_mis[-10:, ip])
    axes[2, ip].set_title(f'part {ip+1}, err = %.2f' % (avg_err))
    axes[2, ip].plot(Log_acc[:, ip], label='acc')
    axes[2, ip].plot(Log_acc[:, ip] + Log_dk[:, ip], label='acc+dk')
    axes[2, ip].plot(
        Log_acc[:, ip] + Log_dk[:, ip] + Log_mis[:, ip],
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
f.savefig(fig_path, dpi=100)

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
        acc_mu, acc_er, acc_mu + dk_mu,
        [n_param], p,
        f, ax,
        title=f'Performance on the TZ task: {cond_name_}',
    )
    fig_path = os.path.join(log_subpath['figs'], f'tz-acc-{cond_name_}.png')
    f.savefig(fig_path, dpi=100)


'''eval the model'''
pad_len_test = 0
p_rm_ob = 0
n_examples_test = 256
fix_cond = None
slience_recall_time = None
scramble = False
epoch_load = epoch_id + 1

np.random.seed(seed_val)
torch.manual_seed(seed_val)
task = SequenceLearning(
    n_param=p.env.n_param, n_branch=p.env.n_branch,
    def_path=p.env.def_path, def_prob=p.env.def_prob, def_tps=p.env.def_tps,
    pad_len=pad_len_test, p_rm_ob_enc=p_rm_ob, p_rm_ob_rcl=p_rm_ob,
    similarity_max=similarity_max, similarity_min=similarity_min,
    similarity_cap_lag=p.n_event_remember,
)

for fix_penalty in np.arange(0, penalty + 1, 2):
    [results, metrics, XY] = run_tz(
        agent, optimizer, task, p, n_examples_test,
        supervised=False, learning=False, get_data=True,
        fix_cond=fix_cond, fix_penalty=fix_penalty,
        slience_recall_time=slience_recall_time, scramble=scramble
    )
    # save the data
    test_params = [fix_penalty, pad_len_test, slience_recall_time]
    test_data_dir, _ = get_test_data_dir(
        log_subpath, epoch_load, test_params)
    test_data_fname = get_test_data_fname(
        n_examples_test, fix_cond, scramble)
    test_data_dict = {'results': results, 'metrics': metrics, 'XY': XY}
    fpath = os.path.join(test_data_dir, test_data_fname)
    pickle_save_dict(test_data_dict, fpath)
