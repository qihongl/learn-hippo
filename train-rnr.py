import os
import time
import torch
import argparse
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import comb
from models import LCALSTM
from data import get_data_rnr, run_exp_rnr
from utils.params import P
from data.ExpRun import process_cache
from utils.io import build_log_path, load_ckpt, save_ckpt, pickle_save_dict
from utils.constants import rnr_log_fnames
plt.switch_backend('agg')

'''learning to tz with a2c
e.g. cmd:
python -u train-rnr.py --exp_name multi-lures --subj_id 0 \
--penalty 4 --n_param 6 --n_hidden 64 \
--epoch_load 150 --n_epoch 100 --n_mvs_rnr 3 \
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
parser.add_argument('--epoch_load', default=50, type=int)
parser.add_argument('--n_epoch', default=1000, type=int)
parser.add_argument('--n_mvs_rnr', default=5, type=int)
parser.add_argument('--log_root', default='../log/', type=str)
args = parser.parse_args()

print(args)
exp_name = args.exp_name
subj_id = args.subj_id
penalty = args.penalty
p_rm_ob_enc = args.p_rm_ob_enc
n_param = args.n_param
n_hidden = args.n_hidden
learning_rate = args.lr
epoch_load = args.epoch_load
n_epoch = args.n_epoch
n_mvs_rnr = args.n_mvs_rnr
log_root = args.log_root

# '''input args'''
# exp_name = 'multi-lures'
# epoch_load = 150
# subj_id = 0
# penalty = 4
# p_rm_ob_enc = 0
# learning_rate = 1e-3
# n_epoch = 1000
# n_mvs_rnr = 3
# log_root = '../log/'
# n_param = 6
# n_hidden = 64

np.random.seed(subj_id)
torch.manual_seed(subj_id)

p = P(
    exp_name=exp_name,
    n_param=n_param, penalty=penalty, n_hidden=n_hidden, lr=learning_rate,
    p_rm_ob_enc=p_rm_ob_enc,
)
p.env.rnr.n_mvs = n_mvs_rnr

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)

# init model
agent = LCALSTM(
    p.net.state_dim, p.net.n_hidden, p.net.n_action,
    recall_func=p.net.recall_func, kernel=p.net.kernel
)
# init optimizer
optimizer_ = optim.Adam(agent.parameters(), lr=p.net.lr)
agent, _ = load_ckpt(epoch_load, log_subpath['ckpts'], agent, optimizer_)

# reinit optimizer
optimizer = optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/2, patience=100, threshold=1e-3, min_lr=1e-8,
    verbose=True
)

'''train on rnr'''
ckpt_template, save_data_fname = rnr_log_fnames(epoch_load, n_epoch)

log_freq = 50
Log_loss_actor = np.zeros(n_epoch,)
Log_loss_critic = np.zeros(n_epoch,)
Log_mistakes = np.zeros(n_epoch,)
# i = 0
for i in range(n_epoch):
    timer0 = time.time()
    D_enc, D_rcl, metadata = get_data_rnr(p.env.rnr.n_mvs, p)
    perfm_metrics, _ = run_exp_rnr(
        agent, optimizer, D_enc, D_rcl, metadata, p, learning=True
    )
    run_time_i = time.time()-timer0
    # log
    [Log_loss_actor[i], Log_loss_critic[i], Log_mistakes[i]] = perfm_metrics
    print('%3d | err: %.2f | L: a: %.2f c: %.2f | time = %.2f sec' % (
        i, Log_mistakes[i], Log_loss_actor[i], Log_loss_critic[i], run_time_i
    ))

    if np.mod(i+1, log_freq) == 0:
        save_ckpt(
            i+1, log_subpath['rnr-ckpts'], agent, optimizer,
            ckpt_template=ckpt_template
        )
    scheduler.step(Log_mistakes[i])


'''plotting, common vars'''
sns.set(style='white', context='talk', palette='colorblind')
alpha = .3
n_se = 2
dpi = 100
c_pal = sns.color_palette()

ws = 10
Log_mistakes_mwd = np.convolve(Log_mistakes, np.ones((ws,))/ws, mode='valid')

f, axes = plt.subplots(2, 1, figsize=(6, 6))
axes[0].plot(Log_mistakes, alpha=alpha, color=c_pal[0])
axes[0].plot(Log_mistakes_mwd, color=c_pal[0])
axes[1].plot(Log_loss_actor, color=c_pal[0], label='actor')
axes[1].plot(Log_loss_critic, color=c_pal[1], label='critic')
axes[1].axhline(0, color='grey', ls='--')
axes[0].set_ylabel('Mistakes')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(frameon=False)
sns.despine()
f.tight_layout()

fig_path = os.path.join(log_subpath['rnr-figs'], 'lcs.png')
f.savefig(fig_path, dpi=dpi, bbox_to_anchor='tight')

'''test'''

n_total_examples = 1024
set_size = int(comb(p.env.rnr.n_mvs, p.env.rnr.n_mvs-1)) * 2
n_sets = n_total_examples // set_size

Y_rcl = np.zeros((n_total_examples, p.env.rnr.event_len, p.net.n_action-1))
Y_hat = np.zeros((n_total_examples, p.env.rnr.event_len, p.net.n_action))
Scalar = np.zeros((n_total_examples, p.env.rnr.event_len, agent.n_ssig))
Vector = np.zeros(
    (n_total_examples, p.env.rnr.event_len, p.net.n_hidden, agent.n_vsig)
)
C, H, M, V = [], [], [], []
Y_encs = []
trial_type_info, memory_ids = [], []

for i in range(n_sets):
    id_l, id_r = np.array([i, (i+1)])*set_size
    # test the model
    D_enc, D_rcl, metadata_i = get_data_rnr(p.env.rnr.n_mvs, p)
    _, cache_i = run_exp_rnr(
        agent, optimizer, D_enc, D_rcl, metadata_i, p, learning=False
    )
    # log
    p_cache_i = process_cache(cache_i)
    [Y_test_, Y_hat_, C_, H_, M_, V_, Vector_, Scalar_, Misc_] = p_cache_i
    [Y_enc] = Misc_
    [trial_type_info_i, memory_ids_i] = metadata_i
    # gather data
    Y_rcl[id_l:id_r] = Y_test_
    Y_hat[id_l:id_r] = Y_hat_
    Vector[id_l:id_r] = Vector_
    Scalar[id_l:id_r] = Scalar_
    C.extend(C_)
    H.extend(H_)
    M.extend(M_)
    V.extend(V_)
    Y_encs.extend(Y_enc)
    trial_type_info.extend(trial_type_info_i)
    memory_ids.extend(memory_ids_i)
memory_ids = np.array(memory_ids)

# save data
model_acts_ = [C, H, V, Scalar, Vector]
Ys_ = [Y_encs, Y_rcl, Y_hat]
data_metadata_ = [memory_ids, trial_type_info]

data_dict = {
    'model_acts': model_acts_,
    'data_labels': Ys_,
    'data_metadata': data_metadata_,
}

output_fpath = os.path.join(log_subpath['rnr-data'], save_data_fname)
pickle_save_dict(data_dict, output_fpath)
