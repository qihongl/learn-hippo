import os
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from models import LCALSTM as Agent
from task import RNR
from models import get_reward, compute_returns, compute_a2c_loss
from analysis import compute_behav_metrics, compute_acc, compute_dk, entropy
from utils.params import P
from utils.utils import to_sqnp
from utils.io import build_log_path, save_ckpt, save_all_params, load_ckpt
from plt_helper import plot_pred_acc_full
from utils.constants import rnr_log_fnames, RNR_COND_DICT
# from sklearn.decomposition.pca import PCA
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
exp_name = 'fulltz-afterrl-ohctx'
subj_id = 0
penalty = 2
supervised_epoch = 300
epoch_load = 500
n_epoch = 500
# n_examples = 256
n_sample = 32
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
# context_dim = n_param + n_branch
context_dim = 10
task = RNR(
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
agent, _ = load_ckpt(epoch_load, log_subpath['ckpts'], agent)
# reinit the agent
optimizer = torch.optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True
)


'''train'''
# take some samples
data_ = task.sample(n_sample)
[X, Y, [rcl_mv_id, cond_id]] = data_
n_data = len(X)
print(f'Number of trials per epoch: {n_data}')

''''''
# prealloc
ckpt_template, save_data_fname = rnr_log_fnames(epoch_load, n_epoch)
log_freq = 50
Log_loss_critic = np.zeros(n_epoch,)
Log_loss_actor = np.zeros(n_epoch,)
Log_return = np.zeros(n_epoch,)
Log_pi_ent = np.zeros(n_epoch,)
Log_acc = np.zeros((n_epoch, task.n_parts))
Log_mis = np.zeros((n_epoch, task.n_parts))
Log_dk = np.zeros((n_epoch, task.n_parts))
Log_cond = np.zeros((n_epoch, n_data))
Log_mem_id = np.zeros((n_epoch, n_data))


epoch_id, i, ip, t = 0, 0, 0, 0
for epoch_id in range(n_epoch):

    time0 = time.time()
    # take sample
    data_ = task.sample(n_sample, permute=True)
    [X, Y, [rcl_mv_id, cond_id]] = data_

    for i in range(n_data):
        # decide when to encode
        enc_times = np.arange(p.net.enc_size-1, task.T_part, p.net.enc_size)
        # prealloc
        log_return = np.zeros(task.n_parts,)
        log_pi_ent = np.zeros(task.n_parts,)
        log_loss_actor = np.zeros(task.n_parts,)
        log_loss_critic = np.zeros(task.n_parts,)
        log_dist_a = np.zeros((n_data, task.n_parts, task.T_part, p.a_dim))
        log_cache = [[None] * task.T_total for _ in range(n_data)]

        agent.init_em_config()
        hc_t = agent.get_init_states()
        loss = 0
        for ip in range(task.n_parts):
            # prealloc
            probs, rewards, values, ents = [], [], [], []

            # only recall for the recall phase
            # TODO remove? this seems to be learnable?
            agent.retrieval_off()
            if ip == task.n_parts-1:
                # flush/turn on recall for the recall phase
                agent.retrieval_on()
                hc_t = agent.get_init_states()

            for t in range(task.T_part):
                # decide if to encode, at time t
                agent.encoding_off()
                if t in enc_times and ip != task.n_parts-1:
                    agent.encoding_on()

                # policy forward
                pi_a_t, v_t, hc_t, cache_t = agent.forward(
                    X[i][ip][t].view(1, 1, -1), hc_t)
                a_t, p_a_t = agent.pick_action(pi_a_t)
                r_t = get_reward(a_t, Y[i][ip][t], p.env.penalty)

                # cache the results for later RL loss computation
                probs.append(p_a_t)
                rewards.append(r_t)
                values.append(v_t)
                ents.append(entropy(pi_a_t))
                # cache results for later analysis
                log_dist_a[i, ip, t, :] = to_sqnp(pi_a_t)
                log_cache[i][t] = cache_t

            # if ip == task.n_parts - 1:
            # compute loss
            returns = compute_returns(rewards)
            loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
            pi_ent = torch.stack(ents).sum()
            loss += loss_actor + loss_critic - pi_ent * p.net.eta

            # update logg after every movie part
            log_return[ip] += torch.stack(rewards).sum().item()
            log_pi_ent[ip] += pi_ent.item()
            log_loss_actor[ip] += loss_actor.item()
            log_loss_critic[ip] += loss_critic.item()

        # update weights after every trial
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # update log after every epoch
    Log_loss_critic[epoch_id] = np.mean(log_loss_critic) / n_data
    Log_loss_actor[epoch_id] = np.mean(log_loss_actor) / n_data
    Log_return[epoch_id] = np.mean(log_return) / n_data
    Log_pi_ent[epoch_id] = np.mean(log_pi_ent) / n_data
    Log_mem_id[epoch_id] = rcl_mv_id
    Log_cond[epoch_id] = cond_id

    # compute behavior metrics for R trials
    r_trials = np.where(cond_id == RNR_COND_DICT.inverse['R'])[0]
    for ip in range(task.n_parts):
        bm_ = compute_behav_metrics(
            Y[r_trials, ip, :, :], log_dist_a[r_trials, ip, :, :], p,
            average_bp=False
        )
        acc_ip_, mis_ip_, dk_ip_ = bm_
        Log_acc[epoch_id, ip] = np.mean(acc_ip_)
        Log_mis[epoch_id, ip] = np.mean(mis_ip_)
        Log_dk[epoch_id, ip] = np.mean(dk_ip_)

    # get log message
    runtime = time.time() - time0
    acc_mu_pts_str = " ".join('%.2f' % i for i in Log_acc[epoch_id])
    dk_mu_pts_str = " ".join('%.2f' % i for i in Log_dk[epoch_id])
    mis_mu_pts_str = " ".join('%.2f' % i for i in Log_mis[epoch_id])
    # print
    msg = '%3d | R: %.2f, acc: %s, dk: %s, mis: %s, ent: %.2f | ' % (
        epoch_id, Log_return[epoch_id],
        acc_mu_pts_str, dk_mu_pts_str, mis_mu_pts_str, Log_pi_ent[epoch_id])
    msg += 'L: a: %.2f c: %.2f| t: %.2fs' % (
        Log_loss_actor[epoch_id], Log_loss_critic[epoch_id], runtime)
    print(msg)

    # update lr scheduler
    neg_pol_score = np.mean(Log_mis[epoch_id]) - np.mean(Log_acc[epoch_id])
    scheduler.step(neg_pol_score)

    # save weights
    if np.mod(epoch_id+1, log_freq) == 0:
        save_ckpt(epoch_id+1, log_subpath['rnr-ckpts'], agent, optimizer,
                  ckpt_template=ckpt_template)


# '''phase 2: view part 1 movies, for half of the movies'''
#
# # set part-id to be the last movie of the i-th trial
# ip = task.n_parts-1
# # turn off encoding, turn on recall
# agent.encoding_off()
# agent.retrieval_on()
# hc_t = agent.get_init_states()
# # prealloc
# probs, rewards, values, ents = [], [], [], []
# # loop over time, for one training example
# for t in range(task.T_part):
#     # forward
#     pi_a_t, v_t, hc_t, cache_t = agent.forward(
#         X[i][ip][t].view(1, 1, -1), hc_t)
#     a_t, p_a_t = agent.pick_action(pi_a_t)
#     r_t = get_reward(a_t, Y[i][ip][t], p.env.penalty)
#     # cache the results for later RL loss computation
#     probs.append(p_a_t)
#     rewards.append(r_t)
#     values.append(v_t)
#     ents.append(entropy(pi_a_t))
#     # cache results for later analysis
#     log_dist_a[i, ip, t, :] = to_sqnp(pi_a_t)
# # compute loss
# returns = compute_returns(rewards)
# loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
# pi_ent = torch.stack(ents).sum()
# loss = loss_actor + loss_critic - pi_ent * p.net.eta
# # update weights
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# # logging
# log_return += r_t
# log_pi_ent += pi_ent.item()
# log_loss_actor += loss_actor.item()
# log_loss_critic += loss_critic.item()
