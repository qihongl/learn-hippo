import os
import time
import torch
import argparse
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from data import get_data_tz, run_exp_tz
from models import LCALSTM
from utils.params import P
from utils.utils import to_sqnp
from utils.constants import TZ_CONDS
from utils.io import build_log_path, save_ckpt, save_all_params
from scipy.stats import sem
from plt_helper import plot_tz_pred_acc
from analysis import compute_predacc, compute_dks, compute_performance_metrics
plt.switch_backend('agg')

'''learning to tz with a2c. e.g. cmd:
python -u train-tz.py --exp_name multi-lures --subj_id 99 \
--penalty 4 --n_param 6 --n_hidden 64 \
--n_epoch 300 --sup_epoch 50 --train_init_state 0 \
--log_root ../log/
'''

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--exp_name', type=str)
parser.add_argument('--subj_id', default=99, type=int)
parser.add_argument('--penalty', default=4, type=int)
parser.add_argument('--p_rm_ob_enc', default=0, type=float)
parser.add_argument('--n_param', default=6, type=int)
parser.add_argument('--n_hidden', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--n_epoch', default=300, type=int)
parser.add_argument('--sup_epoch', default=50, type=int)
parser.add_argument('--n_examples', default=256, type=int)
parser.add_argument('--log_root', default='../log/', type=str)
args = parser.parse_args()
print(args)

# process args
exp_name = args.exp_name
subj_id = args.subj_id
penalty = args.penalty
p_rm_ob_enc = args.p_rm_ob_enc
n_param = args.n_param
n_hidden = args.n_hidden
learning_rate = args.lr
n_examples = args.n_examples
n_epoch = args.n_epoch
supervised_epoch = args.sup_epoch
log_root = args.log_root

# exp_name = 'multi-lures'
# subj_id = 0
# penalty = 4
# p_rm_ob_enc = 0
# supervised_epoch = 50
# n_epoch = 300
# n_examples = 256
# log_root = '../log/'
# n_param = 6
# n_hidden = 64
# learning_rate = 1e-3
rm_ob_probabilistic = True

np.random.seed(subj_id)
torch.manual_seed(subj_id)

p = P(
    exp_name=exp_name,
    n_param=n_param, penalty=penalty, n_hidden=n_hidden, lr=learning_rate,
    p_rm_ob_enc=p_rm_ob_enc, rm_ob_probabilistic=rm_ob_probabilistic,
)
'''init model'''
agent = LCALSTM(
    p.net.state_dim, p.net.n_hidden, p.net.n_action,
    recall_func=p.net.recall_func, kernel=p.net.kernel,
)
optimizer = optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/3, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True
)

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)
# save experiment params initial weights
save_all_params(log_subpath['data'], p, args=None)
save_ckpt(0, log_subpath['ckpts'], agent, optimizer)

'''train'''

# logging params
log_freq = 10
# prealloc
Log_return = np.zeros((n_epoch, ))
Log_mistakes = np.zeros((n_epoch, ))
Log_loss = np.zeros((n_epoch, 3))
Log_corrects_mu = np.zeros((n_epoch, p.env.tz.total_len))
Log_dk_probs_mu = np.zeros((n_epoch, p.env.tz.total_len))

# i, m, t = 0, 0, 0
for i in range(n_epoch):
    timer0 = time.time()
    supervised = i < supervised_epoch

    # get data
    [X, Y], _ = get_data_tz(n_examples, p)
    # train model
    [Log_loss[i, :], Log_return[i], log_adist_i], _ = run_exp_tz(
        agent, optimizer, X, Y, p,
        supervised=supervised, learning=True
    )
    # summarize the performances
    pm_ = compute_performance_metrics(Y, log_adist_i, p)
    Log_corrects_mu[i], Log_dk_probs_mu[i], Log_mistakes[i] = pm_

    # update lr scheduler
    if not supervised:
        scheduler.step(Log_mistakes[i]-Log_return[i])

    # save weights
    if np.mod(i+1, log_freq) == 0:
        save_ckpt(i+1, log_subpath['ckpts'], agent, optimizer)

    # print message
    run_time_i = time.time()-timer0
    print('%3d | R = %.2f, Err = %.2f | L: p = %.2f, v = %.2f, sup = %.2f | time = %.2f sec' % (
        i, Log_return[i], Log_mistakes[i],
        Log_loss[i, 0], Log_loss[i, 1], Log_loss[i, 2],
        run_time_i
    ))


'''plotting, common vars'''
sns.set(style='white', context='talk', palette='colorblind')
event_bounds = p.env.tz.event_ends[:-1]
alpha = .3
n_se = 2
dpi = 100

'''learning curve'''

f, axes = plt.subplots(3, 2, figsize=(10, 7))
f.suptitle('learning curves')

axes[0, 0].plot(Log_return)
axes[0, 0].axhline(0, color='grey', linestyle='--')
axes[0, 0].set_title('final val = %.2f' % Log_return[-1])
axes[0, 0].set_ylabel('Return')

axes[0, 1].plot(Log_mistakes)
axes[0, 1].axhline(0, color='grey', linestyle='--')
axes[0, 1].set_ylabel('Mistakes')
axes[0, 1].set_title('final val = %.2f' % Log_mistakes[-1])

axes[1, 0].plot(Log_loss[:, :2])
axes[1, 0].set_ylabel('Loss, A2C')
axes[1, 0].legend(['actor', 'critic'], frameon=False)
axes[1, 0].axhline(0, color='grey', linestyle='--')

axes[1, 1].plot(Log_loss[:, 2])
axes[1, 1].set_ylabel('Loss, sup')

Log_corrects_mu_ = np.delete(Log_corrects_mu, p.env.tz.event_ends, axis=1)
Log_dk_probs_mu_ = np.delete(Log_dk_probs_mu, p.env.tz.event_ends, axis=1)
sns.heatmap(
    Log_corrects_mu_.T, cmap='RdBu_r', center=p.env.chance,
    ax=axes[2, 0]
)
axes[2, 0].set_ylabel('Corrects')

err_ = np.ones_like(Log_dk_probs_mu_) - Log_corrects_mu_ - Log_dk_probs_mu_
sns.heatmap(err_.T, cmap='viridis', ax=axes[2, 1])
axes[2, 1].set_ylabel('Mistakes')

for i in [0, 1]:
    axes[2, i].hlines(event_bounds, *axes[2, i].get_xlim(), linestyles='--')

sns.despine()
f.tight_layout()
fig_path = os.path.join(log_subpath['figs'], 'lcs.png')
f.savefig(fig_path, dpi=dpi, bbox_to_anchor='tight')

# plot individual results
n_examples_test = 200
for tz_cond in TZ_CONDS:
    # make test set
    [X, Y], _ = get_data_tz(n_examples_test, p)
    # run the model on the text set
    [log_loss_i, log_return_i, log_adist_i], _ = run_exp_tz(
        agent, optimizer, X, Y, p,
        cond=tz_cond, learning=False,
    )

    '''plot accuracy'''
    # compuet corrects and dk indicator mats
    Y_np = to_sqnp(Y)
    corrects = compute_predacc(Y_np, log_adist_i)
    dks = compute_dks(log_adist_i)
    # trim event end points
    corrects_ = np.delete(corrects, p.env.tz.event_ends, axis=1)
    dks_ = np.delete(dks, p.env.tz.event_ends, axis=1)

    # compute mus
    pa_mu = np.mean(corrects_, axis=0)
    pa_er = sem(corrects_, axis=0) * n_se
    dk_probs_mu = np.mean(dks_, axis=0)
    dk_probs_er = sem(dks_, axis=0) * n_se
    #
    pa_or_dk_mu = pa_mu + dk_probs_mu

    f, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
    plot_tz_pred_acc(pa_mu, pa_er, pa_or_dk_mu, event_bounds, p, f, ax)
    fig_path = os.path.join(log_subpath['figs'], 'tz-%s.png' % tz_cond)
    f.savefig(fig_path, dpi=dpi, bbox_to_anchor='tight')
