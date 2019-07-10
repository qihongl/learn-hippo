'''eval a bunch of models for tz
'''
import numpy as np
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from data import get_data_tz, run_exp_tz
from models import LCALSTM
from models import pick_action
from analysis import compute_predacc, compute_dks
from utils.utils import to_np
from utils.params import P
from utils.io import build_log_path, load_ckpt
from scipy.stats import sem
from plt_helper import plot_pred_acc_full

sns.set(style='white', context='talk', palette='colorblind')
alpha = .3
n_se = 2
dpi = 100

'''input args'''
log_root = '../log/'
exp_name = 'testing'
epoch_load = 300
penalty = 4

n_param = 6
n_branch = 3
n_hidden = 64
learning_rate = 1e-3
# penaltys = [0, 4]
# for penalty in penaltys:
subj_ids = [99]
# subj_ids = range(10)
# n_subj = len(subj_ids)

'''fixed params'''
# log params
p = P(
    exp_name=exp_name,
    n_param=n_param, penalty=penalty, n_hidden=n_hidden, lr=learning_rate,
)
agent = LCALSTM(
    p.net.state_dim, p.net.n_hidden, p.net.n_action,
    recall_func=p.net.recall_func, kernel=p.net.kernel,
)
optimizer = optim.Adam(agent.parameters(), lr=p.net.lr)


# # cache for all subjs
# log_pa = np.zeros((n_subj, p.env.tz.n_mvs * n_param))
# log_pa_or_dk = np.zeros((n_subj, p.env.tz.n_mvs * n_param))
# supervised = False
# tz_cond = 'RM'

i_s, subj_id = 0, subj_ids[0]
# for i_s, subj_id in enumerate(subj_ids):

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)
agent, _ = load_ckpt(epoch_load, log_subpath['ckpts'], agent, optimizer)

# make test set
tz_cond = 'RM'
n_examples_test = 300
[X, Y], _ = get_data_tz(n_examples_test, p)
perfm_metrics, cache_return = run_exp_tz(
    agent, optimizer, X, Y, p,
    cond=tz_cond, supervised=False,
    learning=False,
)
log_loss_i, log_return_i, log_adist_i = perfm_metrics
[Y, log_adist, C, M, V, Vector, Scalar, misc_return] = cache_return


p.env.event_len

#
# # run the model on the text set
# log_adist = np.zeros((n_examples_test, total_event_len, n_action))
# for m in range(n_examples_test):
#     hc_t = get_init_states(p.net.n_hidden)
#     for t in range(total_event_len):
#         # get current state to predict action value
#         action_dist_t, v_t, hc_t, cache_t = agent(
#             X[m][t].view(1, 1, -1), hc_t)
#         a_t, log_prob_a_t = pick_action(action_dist_t)
#         # log data
#         log_adist[m, t, :] = np.squeeze(to_np(action_dist_t))

'''plot accuracy'''
# compuet corrects and dk indicator mats
Y_np = np.squeeze(to_np(Y))
corrects = compute_predacc(Y_np, log_adist)
dks = compute_dks(log_adist)
# trim event end points
corrects_ = np.delete(corrects, p.env.tz.event_ends, axis=1)
dks_ = np.delete(dks, p.env.tz.event_ends, axis=1)

# compute mus
pa_mu = np.mean(corrects_, axis=0)
pa_er = sem(corrects_, axis=0)*n_se
dk_probs_mu = np.mean(dks_, axis=0)
dk_probs_er = sem(dks_, axis=0) * n_se
#
pa_or_dk_mu = pa_mu + dk_probs_mu

# # plot individual results
# f, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
# plot_pred_acc_full(
#     pa_mu, pa_er, pa_or_dk_mu, event_bounds, p,
#     f, ax
# )
# fig_path = os.path.join(log_subpath['figs'], 'fineval.png')
# f.savefig(fig_path, dpi=dpi, bbox_to_anchor='tight')

# log data for all subjects
log_pa[i_s, :] = pa_mu
log_pa_or_dk[i_s, :] = pa_or_dk_mu

# out of the loop over subjects
log_pa_mu = np.mean(log_pa, axis=0)
log_pa_se = sem(log_pa, axis=0) * n_se
log_pa_or_dk_mu = np.mean(log_pa_or_dk, axis=0)

# plot the group results
f, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
plot_pred_acc_full(
    log_pa_mu, log_pa_se, log_pa_or_dk_mu, event_bounds, p,
    f, ax
)

fig_path = os.path.join('./figs', f'fineval-p{penalty}.png')
f.savefig(fig_path, dpi=dpi, bbox_to_anchor='tight')
