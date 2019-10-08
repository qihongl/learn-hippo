import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.LCALSTM_v9 import LCALSTM as Agent
from task import SequenceLearning
from exp_aba import run_aba
from analysis import compute_behav_metrics, compute_acc, compute_dk
from vis import plot_pred_acc_full
from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, save_ckpt, save_all_params,  \
    pickle_save_dict, get_test_data_dir, get_test_data_fname

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
parser.add_argument('--def_prob', default=None, type=float)
parser.add_argument('--enc_size', default=None, type=int)
parser.add_argument('--penalty', default=4, type=int)
parser.add_argument('--penalty_random', default=0, type=int)
parser.add_argument('--penalty_discrete', default=1, type=int)
parser.add_argument('--penalty_onehot', default=0, type=int)
parser.add_argument('--normalize_return', default=1, type=int)
parser.add_argument('--p_rm_ob_enc', default=0, type=float)
parser.add_argument('--p_rm_ob_rcl', default=0, type=float)
parser.add_argument('--similarity_cap', default=None, type=float)
parser.add_argument('--n_hidden', default=64, type=int)
parser.add_argument('--n_hidden_dec', default=32, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--eta', default=0.1, type=float)
parser.add_argument('--n_event_remember', default=4, type=int)
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
enc_size = args.enc_size
penalty = args.penalty
penalty_random = args.penalty_random
penalty_discrete = args.penalty_discrete
penalty_onehot = args.penalty_onehot
normalize_return = args.normalize_return
p_rm_ob_enc = args.p_rm_ob_enc
p_rm_ob_rcl = args.p_rm_ob_rcl
similarity_cap = args.similarity_cap
n_hidden = args.n_hidden
n_hidden_dec = args.n_hidden_dec
learning_rate = args.lr
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
    exp_name=exp_name,
    sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    def_prob=def_prob,
    enc_size=enc_size, n_event_remember=n_event_remember,
    penalty=penalty, penalty_random=penalty_random,
    penalty_discrete=penalty_discrete, penalty_onehot=penalty_onehot,
    normalize_return=normalize_return,
    p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
    n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
    lr=learning_rate, eta=eta,
)
# init env
n_parts = 3
task = SequenceLearning(
    n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=p.env.pad_len,
    p_rm_ob_enc=p.env.p_rm_ob_enc, p_rm_ob_rcl=p.env.p_rm_ob_rcl,
    similarity_cap_lag=p.n_event_remember, similarity_cap=similarity_cap,
    n_parts=n_parts
)
# init agent
agent = Agent(
    input_dim=task.x_dim+p.extra_x_dim, output_dim=p.a_dim,
    rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
    dict_len=p.net.dict_len
)

optimizer = torch.optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True)

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)


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
Log_cond = np.zeros((n_epoch, n_examples//2))

# epoch_id, i, t = 0, 0, 0
epoch_id = 0
for epoch_id in np.arange(epoch_id, n_epoch):
    time0 = time.time()
    [results, metrics] = run_aba(
        agent, optimizer, task, p, n_examples,
        supervised=False, fix_cond='DM', learning=True, get_cache=False,
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
    neg_pol_score = np.mean(Log_mis[epoch_id]) - np.mean(Log_acc[epoch_id])
    scheduler.step(neg_pol_score)

    # save weights
    if np.mod(epoch_id+1, log_freq) == 0:
        save_ckpt(epoch_id+1, log_subpath['ckpts'], agent, optimizer)
