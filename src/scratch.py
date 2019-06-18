import os
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from models import LCALSTM
from task import TwilightZone
from models import get_reward, compute_returns, compute_a2c_loss
from analysis import compute_acc, compute_dk, average_by_part, entropy
from utils.params import P
from utils.utils import to_sqnp
from utils.io import build_log_path, save_ckpt, save_all_params, load_ckpt
# from data import get_data_tz, run_exp_tz
# from utils.constants import TZ_CONDS
from scipy.stats import sem
# from plt_helper import plot_tz_pred_acc
# from analysis.utils import get_tps_for_ith_part
# plt.switch_backend('agg')
from utils.constants import TZ_CONDS

'''learning to tz with a2c. e.g. cmd:
python -u train-tz.py --exp_name multi-lures --subj_id 99 \
--penalty 4 --n_param 6 --n_hidden 64 \
--n_epoch 300 --sup_epoch 50 --train_init_state 0 \
--log_root ../log/
'''
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-n', '--exp_name', type=str)
# parser.add_argument('--subj_id', default=99, type=int)
# parser.add_argument('--penalty', default=4, type=int)
# parser.add_argument('--p_rm_ob_enc', default=0, type=float)
# parser.add_argument('--n_param', default=6, type=int)
# parser.add_argument('--n_hidden', default=64, type=int)
# parser.add_argument('--lr', default=1e-3, type=float)
# parser.add_argument('--n_epoch', default=300, type=int)
# parser.add_argument('--sup_epoch', default=50, type=int)
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
# n_param = args.n_param
# n_hidden = args.n_hidden
# learning_rate = args.lr
# n_examples = args.n_examples
# n_epoch = args.n_epoch
# supervised_epoch = args.sup_epoch
# log_root = args.log_root

exp_name = 'test-linear'
subj_id = 0
penalty = 4
p_rm_ob_enc = 0
supervised_epoch = 50
n_epoch = 200
n_examples = 256
log_root = '../log/'
n_param = 6
n_hidden = 64
learning_rate = 1e-3
gamma = 0
eta = .1

np.random.seed(subj_id)
torch.manual_seed(subj_id)

'''init'''
p = P(
    exp_name=exp_name,
    n_param=n_param, penalty=penalty, n_hidden=n_hidden, lr=learning_rate,
    p_rm_ob_enc=p_rm_ob_enc,
)

# init agent
agent = LCALSTM(
    p.net.x_dim, p.net.n_hidden, p.net.a_dim,
    recall_func=p.net.recall_func, kernel=p.net.kernel,
)
optimizer = torch.optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/3, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True
)
# init env
tz = TwilightZone(
    p.env.n_param, p.env.n_branch,
    p_rm_ob_enc=p.env.p_rm_ob_enc
)

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)
# save experiment params initial weights
save_all_params(log_subpath['data'], p, args=None)
save_ckpt(0, log_subpath['ckpts'], agent, optimizer)

# load model
epoch_load = 50
agent, optimizer = load_ckpt(epoch_load, log_subpath['ckpts'], agent, optimizer)
epoch_id = epoch_load

'''task definition'''


def allow_dk(t, tz_cond, t_allowed):
    if t < t_allowed or tz_cond is 'NM':
        return True
    return False


def pick_condition(supervised, p):
    if supervised:
        tz_cond = 'RM'
    else:
        tz_cond = np.random.choice(TZ_CONDS, p=p.env.tz.p_cond)
    return tz_cond


def set_encoding_flag(t, enc_times, agent):
    if t in enc_times:
        agent.encoding_on()
    else:
        agent.encoding_off()


def tz_cond_manipulation(tz_cond, t, event_bond, p, hc_t, agent, n_lures=1):
    '''condition specific manipulation
    such as flushing, insert lure, etc.
    '''
    if t == event_bond:
        agent.retrieval_on()
        if tz_cond == 'DM':
            # RM: has EM, no WM
            hc_t = agent.get_init_states()
            agent.add_simple_lures(n_lures)
        elif tz_cond == 'NM':
            # RM: no WM, EM
            hc_t = agent.get_init_states()
            agent.flush_episodic_memory()
            agent.add_simple_lures(n_lures+1)
        elif tz_cond == 'RM':
            # RM: has WM, EM
            agent.add_simple_lures(n_lures)
        else:
            raise ValueError('unrecog tz condition')
    else:
        pass
    return hc_t


log_freq = 10
Log_loss_actor = np.zeros(n_epoch,)
Log_loss_critic = np.zeros(n_epoch,)
Log_loss_sup = np.zeros(n_epoch,)
Log_pi_ent = np.zeros(n_epoch,)
Log_mistakes = np.zeros(n_epoch,)
Log_return = np.zeros(n_epoch,)

cond = None
learning = True
# epoch_id, i, t = 0, 0, 0

# epoch_id = 0
for epoch_id in np.arange(epoch_id, n_epoch):
    time0 = time.time()

    # sample data
    X, Y = tz.sample(n_examples, to_torch=True)
    # training objective
    supervised = epoch_id < supervised_epoch
    # pick condition
    if cond is None:
        tz_cond = pick_condition(supervised, p)
    else:
        tz_cond = cond

    tz_cond = 'NM'

    # logger
    log_return, log_pi_ent = 0, 0
    log_loss_sup, log_loss_actor, log_loss_critic = 0, 0, 0
    log_dist_a = np.zeros((n_examples, tz.T_total, p.a_dim))

    for i in range(n_examples):
        # pg calculation cache
        probs, rewards, values, ents = [], [], [], []
        # init model wm and em
        hc_t = agent.get_init_states()
        agent.init_em_config()

        for t in range(tz.T_total):

            # whether to encode
            if not supervised:
                set_encoding_flag(t, [p.env.tz.event_ends[0]], agent)

            # get next state and action target
            y_t_targ = torch.squeeze(Y[i][t])
            a_t_targ = torch.argmax(y_t_targ)
            # forward
            pi_a_t, v_t, hc_t, cache_t = agent(X[i][t].view(1, 1, -1), hc_t)
            a_t, p_a_t = agent.pick_action(pi_a_t)
            r_t = get_reward(
                a_t, a_t_targ, p.dk_id, p.env.penalty,
                allow_dk=allow_dk(t, tz_cond, p.env.tz.event_ends[0])
            )
            # cache the results for later RL loss computation
            probs.append(p_a_t)
            rewards.append(r_t)
            values.append(v_t)
            ents.append(entropy(pi_a_t))
            # cache results for later analysis
            log_dist_a[i, t, :] = to_sqnp(pi_a_t)

            # compute supervised loss
            yhat_t = torch.squeeze(pi_a_t)[:-1]
            loss_sup_it = F.mse_loss(yhat_t, y_t_targ)
            log_loss_sup += loss_sup_it.item() / (tz.T_total*n_examples)
            if learning and supervised:
                optimizer.zero_grad()
                loss_sup_it.backward(retain_graph=True)
                optimizer.step()

            if not supervised:
                # update WM/EM bsaed on the condition
                hc_t = tz_cond_manipulation(
                    tz_cond, t, p.env.tz.event_ends[0], p, hc_t, agent)

        # compute RL loss
        returns = compute_returns(rewards, gamma=gamma)
        loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
        pi_ent = torch.stack(ents).sum()
        if learning and not supervised:
            loss_rl = loss_actor + loss_critic - pi_ent * eta
            optimizer.zero_grad()
            loss_rl.backward()
            optimizer.step()

        # after every event sequence, log stuff
        log_pi_ent += pi_ent.item()/(n_examples * tz.T_total)
        log_return += torch.stack(rewards).sum().item()/n_examples
        log_loss_actor += loss_actor.item()/n_examples
        log_loss_critic += loss_critic.item()/n_examples

    # log
    Log_pi_ent[epoch_id] = log_pi_ent
    Log_return[epoch_id] = log_return
    Log_loss_sup[epoch_id] = log_loss_sup
    Log_loss_actor[epoch_id] = log_loss_actor
    Log_loss_critic[epoch_id] = log_loss_critic

    # update lr scheduler
    if not supervised:
        scheduler.step(Log_return[epoch_id])

    # compute performance
    acc_mu_ = compute_acc(Y, log_dist_a)
    dk_mu_ = compute_dk(log_dist_a)
    # split by movie parts
    acc_mu_parts = average_by_part(acc_mu_, p)
    dk_mu_parts = average_by_part(dk_mu_, p)

    # log message
    runtime = time.time() - time0
    acc_mu_parts_str = " ".join('%.2f' % i for i in acc_mu_parts)
    dk_mu_parts_str = " ".join('%.2f' % i for i in dk_mu_parts)
    print('%3d | R: %.2f, acc: %s, dk: %s, ent: %.2f | L: a: %.2f c: %.2f, s: %.2f | t: %.2f s' % (
        epoch_id, Log_return[epoch_id], acc_mu_parts_str, dk_mu_parts_str,
        Log_pi_ent[epoch_id],
        Log_loss_actor[epoch_id], Log_loss_critic[epoch_id],
        Log_loss_sup[epoch_id], runtime
    ))

    # save weights
    if np.mod(epoch_id+1, log_freq) == 0:
        save_ckpt(epoch_id+1, log_subpath['ckpts'], agent, optimizer)

# np.mean(np.mean(log_dist_a, axis=0), axis=0)

# plt.plot(acc_mu_)

for name, w, in agent.named_parameters():
    print(name)
    print(w)
