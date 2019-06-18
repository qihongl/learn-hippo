import numpy as np
import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
# from data.utils import sample_rand_path
from data import get_data_tz
from models import LCALSTM
from models import pick_action, compute_returns, get_reward
from models.utils import get_init_states, smart_init
from analysis import compute_predacc, get_event_ends, compute_dks
from utils.utils import to_sqnp
from utils.constants import TZ_CONDS
from utils.params import P
from utils.io import build_log_path, save_ckpt, save_all_params
from scipy.stats import sem
from plt_helper import plot_state_prediction_acc


'''learning to tz with a2c
e.g. cmd:
python train-tz.py --subj_id 0 --penalty 5 \
--log_root '../log/'
'''


# parser = argparse.ArgumentParser()
# parser.add_argument('--subj_id', type=int)
# parser.add_argument('--penalty', type=int)
# parser.add_argument('--epoch', default=500, type=int)
# parser.add_argument('--sup_epoch', default=100, type=int)
# parser.add_argument('--log_root', type=str)
# args = parser.parse_args()
# print(args)
#
# # process args
# subj_id = args.subj_id
# penalty = args.penalty
# n_epoch = args.sup_epoch
# supervised_epoch = args.sup_epoch
# LOG_ROOT = args.log_root
p_TZ_CONDS = [.6, .3, .1]

n_epoch = 400
subj_id = 103
supervised_epoch = 50
no_memory_epoch = 75
penalty = 4

LOG_ROOT = '../log/'
n_examples = 64

# np.random.seed(subj_id)
# torch.manual_seed(subj_id)

'''fixed params'''
# event parameters
n_param = 6
n_branch = 3
pad_len = 1
event_len = n_param + pad_len
n_movies_rnr = 5
n_hidden = 80
learning_rate = 1e-3
# log params
p = P(
    n_param=n_param, n_branch=n_branch, pad_len=pad_len,
    n_mvs=n_movies_rnr,
    penalty=penalty, n_hidden=n_hidden, lr=learning_rate
)

# init model
agent = LCALSTM(
    p.net.state_dim, p.net.n_hidden, p.net.n_action,
    recall_func=p.net.recall_func, kernel=p.net.kernel
)
agent = smart_init(agent)
# init optimizer
optimizer = optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/3, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True
)

# create logging dirs
log_path, log_subpath = build_log_path(subj_id, p)
# save experiment params initial weights
save_all_params(log_subpath['data'], p, args=None)
save_ckpt(0, log_subpath['ckpts'], agent, optimizer)

'''train
'''


def compute_a2c_loss(log_probs, values, returns):
    policy_losses, value_losses = [], []
    for log_prob_t, v_t, R_t in zip(log_probs, values, returns):
        advantage = R_t - v_t.item()
        # advantage = R_t
        policy_losses.append(-log_prob_t * advantage)
        value_losses.append(F.smooth_l1_loss(v_t, torch.tensor([R_t])))
    loss_policy = torch.stack(policy_losses).sum()
    loss_value = torch.stack(value_losses).sum()
    return loss_policy, loss_value


# logging params
log_freq = 10

# training params
n_repeats_tz = 2
total_event_len = event_len * n_repeats_tz
event_ends = get_event_ends(event_len, n_repeats_tz)

# prealloc
Log_return = np.zeros((n_epoch, ))
Log_mistakes = np.zeros((n_epoch, ))
Log_loss = np.zeros((n_epoch, 3))
Log_corrects_mu = np.zeros((n_epoch, total_event_len))
Log_dk_probs_mu = np.zeros((n_epoch, total_event_len))

# i, m, t = 0, 0, 0
# tz_cond = 'DM'

learning = True
for i in range(n_epoch):

    timer0 = time.time()
    [X, Y], _ = get_data_tz(n_examples, n_repeats_tz, p)
    _, total_event_len, _, _ = X.size()
    # temp log
    Log_adists_ = np.zeros((n_examples, total_event_len, p.net.n_action))

    for m in range(n_examples):

        # if i > no_memory_epoch:
        #     tz_cond = np.random.choice(TZ_CONDS, p=p_TZ_CONDS)
        # else:
        #     tz_cond = 'RM'
        tz_cond = 'RM'
        # pg calculation cache
        log_probs, rewards, values = [], [], []
        # logging
        action_dists = []
        loss_sup = 0

        agent.flush_episodic_memory()
        agent.encoding_off()
        if i > no_memory_epoch:
            agent.retrieval_on()
        else:
            agent.retrieval_off()

        hc_t = get_init_states(p.net.n_hidden)
        for t in range(total_event_len):

            if t < event_ends[0] or tz_cond is 'NM':
                allow_dk = True
            else:
                allow_dk = False

            if i > no_memory_epoch:
                if t == event_ends[0]:
                    agent.encoding_on()
                else:
                    agent.encoding_off()

            # get next state and action target
            y_t_targ = torch.squeeze(Y[m][t])
            a_t_targ = torch.argmax(y_t_targ)
            # forward
            action_dist_t, v_t, hc_t, cache_t = agent(
                X[m][t].view(1, 1, -1), hc_t, beta=1)
            a_t, log_prob_a_t = pick_action(action_dist_t)
            r_t = get_reward(
                a_t, a_t_targ, p.net.dk_id, p.env.penalty, allow_dk=allow_dk)
            # cache the results for later RL loss computation
            log_probs.append(log_prob_a_t)
            rewards.append(r_t)
            values.append(v_t)
            # cache results for later analysis
            action_dists.append(to_sqnp(action_dist_t))

            # compute supervised loss
            yhat_t = torch.squeeze(action_dist_t)[:p.net.n_action-1]
            # no cost for the last (dummy) event
            if t not in event_ends:
                loss_sup += F.mse_loss(yhat_t, y_t_targ)
            # condition specific
            if i > no_memory_epoch:
                if t == event_ends[0]:
                    agent.retrieval_on()
                    lm = torch.randn(1, 1, p.net.n_hidden)
                    if tz_cond == 'DM':
                        hc_t = get_init_states(p.net.n_hidden)
                    elif tz_cond == 'NM':
                        hc_t = get_init_states(p.net.n_hidden)
                        agent.flush_episodic_memory()
                        agent.inject_memories([lm], [lm])
                    elif tz_cond == 'RM':
                        pass
                    else:
                        raise ValueError('unrecog tz condition')
                else:
                    pass

        # calculate sample return
        returns = compute_returns(rewards)
        # compute loss
        loss_actor, loss_critic = compute_a2c_loss(
            log_probs, values, returns)

        if learning:
            # switch loss
            if i > supervised_epoch:
                loss = loss_actor + loss_critic
            else:
                loss = loss_sup

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # after every event sequence, log stuff
        Log_return[i] += torch.stack(rewards).sum().item()/n_examples
        Log_loss[i, :] += np.array(
            [loss_actor.item(), loss_critic.item(), loss_sup.item()]
        )/n_examples
        # compute performance metrics
        Log_adists_[m] = np.stack(action_dists)

        # compute correct rate
        corrects = compute_predacc(to_sqnp(Y), Log_adists_)
        dks = compute_dks(Log_adists_)
        # compute mus
        corrects_mu_ = np.mean(corrects, axis=0)
        dk_probs_mu_ = np.mean(dks, axis=0)
        corrects_mu__ = np.delete(corrects_mu_, event_ends)
        dk_probs_mu__ = np.delete(dk_probs_mu_, event_ends)
        mistakes_ = np.ones_like(corrects_mu__) - \
            corrects_mu__ - dk_probs_mu__
        # log performance stats
        Log_corrects_mu[i] = corrects_mu_
        Log_dk_probs_mu[i] = dk_probs_mu_
        Log_mistakes[i] = np.sum(mistakes_)

    # after every episode...

    # update lr scheduler
    if learning and i > supervised_epoch:
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
chance = 1/n_branch
event_bounds = event_ends[:-1]
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

Log_corrects_mu_ = np.delete(Log_corrects_mu, event_ends, axis=1)
Log_dk_probs_mu_ = np.delete(Log_dk_probs_mu, event_ends, axis=1)
sns.heatmap(Log_corrects_mu_.T, cmap='RdBu_r', center=chance, ax=axes[2, 0])
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
tz_cond = 'RM'
# make test set
[X, Y], _ = get_data_tz(n_examples_test, n_repeats_tz, p)
_, total_event_len, _, _ = X.size()

# run the model on the text set
log_adist = np.zeros((n_examples_test, total_event_len, p.net.n_action))
for m in range(n_examples_test):

    agent.flush_episodic_memory()
    agent.encoding_off()
    agent.retrieval_on()

    hc_t = get_init_states(p.net.n_hidden)
    for t in range(total_event_len):
        if t == event_ends[0]:
            agent.encoding_on()
        else:
            agent.encoding_off()
        # get current state to predict action value
        action_dist_t, v_t, hc_t, cache_t = agent(
            X[m][t].view(1, 1, -1), hc_t)
        a_t, log_prob_a_t = pick_action(action_dist_t)
        # log data
        log_adist[m, t, :] = to_sqnp(action_dist_t)

        # condition specific
        if t == event_ends[0]:
            agent.retrieval_on()
            if tz_cond == 'DM':
                hc_t = get_init_states(p.net.n_hidden)
            elif tz_cond == 'NM':
                hc_t = get_init_states(p.net.n_hidden)
                agent.flush_episodic_memory()
                lm = torch.randn(1, 1, p.net.n_hidden)
                agent.inject_memories([lm], [lm])
            elif tz_cond == 'RM':
                pass
            else:
                raise ValueError('unrecog tz condition')
        else:
            pass

'''plot accuracy'''
# compuet corrects and dk indicator mats
Y_np = to_sqnp(Y)
corrects = compute_predacc(Y_np, log_adist)
dks = compute_dks(log_adist)
# trim event end points
corrects_ = np.delete(corrects, event_ends, axis=1)
dks_ = np.delete(dks, event_ends, axis=1)

# compute mus
pa_mu = np.mean(corrects_, axis=0)
pa_er = sem(corrects_, axis=0)*n_se
dk_probs_mu = np.mean(dks_, axis=0)
dk_probs_er = sem(dks_, axis=0) * n_se
#
pa_or_dk_mu = pa_mu + dk_probs_mu

f, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
plot_state_prediction_acc(
    pa_mu, pa_er, pa_or_dk_mu, event_bounds, p,
    f, ax
)
fig_path = os.path.join(log_subpath['figs'], 'fineval.png')
f.savefig(fig_path, dpi=dpi, bbox_to_anchor='tight')
