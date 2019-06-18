import os
import time
import torch
import argparse
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from models import LCALSTM
from utils.params import P
from utils.utils import to_sqnp
# from data import get_data_tz, run_exp_tz
# from utils.constants import TZ_CONDS
# from utils.io import build_log_path, save_ckpt, save_all_params
# from scipy.stats import sem
# from plt_helper import plot_tz_pred_acc
# from analysis import compute_predacc, compute_dks, compute_performance_metrics
from task import TwilightZone
from models import get_reward, compute_returns, compute_a2c_loss
plt.switch_backend('agg')

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

exp_name = 'multi-lures'
subj_id = 0
penalty = 4
p_rm_ob_enc = 0
supervised_epoch = 50
n_epoch = 300
n_examples = 256
log_root = '../log/'
n_param = 6
n_hidden = 64
learning_rate = 1e-3
rm_ob_probabilistic = True

np.random.seed(subj_id)
torch.manual_seed(subj_id)

p = P(
    exp_name=exp_name,
    n_param=n_param, penalty=penalty, n_hidden=n_hidden, lr=learning_rate,
    p_rm_ob_enc=p_rm_ob_enc, rm_ob_probabilistic=rm_ob_probabilistic,
)

'''init'''
# init env
tz = TwilightZone(
    n_param, p.env.n_branch, p_rm_ob_enc=p_rm_ob_enc
)

# init agent
n_action = tz.y_dim+1
agent = LCALSTM(
    tz.x_dim, p.net.n_hidden, n_action,
    recall_func=p.net.recall_func, kernel=p.net.kernel,
)
optimizer = optim.Adam(agent.parameters(), lr=p.net.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/3, patience=30, threshold=1e-3, min_lr=1e-8,
    verbose=True
)

'''task definition'''


def allow_dk(t, tz_cond, t_allowed):
    if t < t_allowed or tz_cond is 'NM':
        return True
    return False


Log_loss_actor = np.zeros(n_epoch,)
Log_loss_critic = np.zeros(n_epoch,)
Log_mistakes = np.zeros(n_epoch,)

epoch_id, i, t = 0, 0, 0

for epoch_id in range(n_epoch):

    X, Y = tz.sample(n_examples, to_torch=True)
    # logger
    log_adist = np.zeros((n_examples, tz.T_total, n_action))
    log_return = 0
    log_loss = np.zeros(3,)
    tz_cond = 'RM'
    supervised = epoch_id < supervised_epoch

    for i in range(n_examples):

        # pg calculation cache
        probs, rewards, values = [], [], []
        # logging
        action_dists = []
        loss_sup = 0

        # init model wm and em
        hc_t = agent.get_init_states()
        agent.init_em_config()
        for t in range(tz.T_total):

            # get next state and action target
            y_t_targ = torch.squeeze(Y[i][t])
            a_t_targ = torch.argmax(y_t_targ)
            # forward
            action_dist_t, v_t, hc_t, cache_t = agent(
                X[i][t].view(1, 1, -1), hc_t, beta=1)
            a_t, prob_a_t = agent.pick_action(action_dist_t)
            r_t = get_reward(
                a_t, a_t_targ, n_action, p.env.penalty,
                allow_dk=allow_dk(t, tz_cond, p.env.tz.event_ends[0])
            )
            # cache the results for later RL loss computation
            probs.append(prob_a_t)
            rewards.append(r_t)
            values.append(v_t)
            # cache results for later analysis
            action_dists.append(to_sqnp(action_dist_t))

            # compute supervised loss
            yhat_t = torch.squeeze(action_dist_t)[:n_action-1]
            # no cost for the last (dummy) event
            if t not in p.env.tz.event_ends:
                loss_sup += torch.nn.functional.mse_loss(yhat_t, y_t_targ)

        # calculate sample return
        returns = compute_returns(rewards)
        # compute loss
        loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
        loss = loss_actor + loss_critic

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # after every event sequence, log stuff
        log_adist[i] = np.stack(action_dists)
        log_return += torch.stack(rewards).sum().item()/n_examples
        log_loss += np.array(
            [loss_actor.item(), loss_critic.item(), loss_sup.item()]
        )/n_examples
