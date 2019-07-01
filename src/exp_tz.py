import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import to_sqnp
from analysis import entropy
from models import get_reward, compute_returns, compute_a2c_loss


def run_tz(
        agent, optimizer, task, p, n_examples, supervised,
        cond=None, learning=True
):
    # sample data
    X, Y = task.sample(n_examples, to_torch=True)
    # logger
    log_return, log_pi_ent = 0, 0
    log_loss_sup, log_loss_actor, log_loss_critic = 0, 0, 0
    log_cond = np.zeros(n_examples,)
    log_dist_a = np.zeros((n_examples, task.T_total, p.a_dim))
    log_cache = [[None] * task.T_total for _ in range(n_examples)]
    for i in range(n_examples):
        # pick a condition
        cond_i = pick_condition(p, rm_only=supervised, fix_cond=cond)
        # init model wm and em
        hc_t = agent.get_init_states()
        agent.retrieval_off()
        agent.encoding_off()

        # pg calculation cache
        loss_sup = 0
        probs, rewards, values, ents = [], [], [], []
        for t in range(task.T_total):
            # whether to encode
            if not supervised:
                set_encoding_flag(t, [task.event_ends[0]], cond_i, agent)
            # forward
            pi_a_t, v_t, hc_t, cache_t = agent.forward(
                X[i][t].view(1, 1, -1), hc_t)

            # after delay period, compute loss
            if np.mod(t, task.T_part) >= task.pad_len:
                a_t, p_a_t = agent.pick_action(pi_a_t)
                r_t = get_reward(a_t, Y[i][t], p.env.penalty)
                # cache the results for later RL loss computation
                rewards.append(r_t)
                values.append(v_t)
                probs.append(p_a_t)
                ents.append(entropy(pi_a_t))
                # compute supervised loss
                yhat_t = torch.squeeze(pi_a_t)[:-1]
                loss_sup += F.mse_loss(yhat_t, Y[i][t])

            # cache results for later analysis
            log_dist_a[i, t, :] = to_sqnp(pi_a_t)
            log_cache[i][t] = cache_t

            if not supervised:
                # update WM/EM bsaed on the condition
                hc_t = cond_manipulation(
                    cond_i, t, task.event_ends[0], hc_t, agent)

        # compute RL loss
        returns = compute_returns(rewards, normalize=True)
        loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
        pi_ent = torch.stack(ents).sum()
        # if learning and not supervised
        if learning:
            if supervised:
                loss = loss_sup
            else:
                loss = loss_actor + loss_critic - pi_ent * p.net.eta
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
            optimizer.step()

        # after every event sequence, log stuff
        log_loss_sup += loss_sup / n_examples
        log_pi_ent += pi_ent.item() / n_examples
        log_return += torch.stack(rewards).sum().item()/n_examples
        log_loss_actor += loss_actor.item()/n_examples
        log_loss_critic += loss_critic.item()/n_examples
        log_cond[i] = p.env.tz.cond_dict.inverse[cond_i]

    # return cache
    results = [log_dist_a, Y, log_cache, log_cond]
    metrics = [log_loss_sup, log_loss_actor, log_loss_critic,
               log_return, log_pi_ent]
    out = [results, metrics]
    return out


def pick_condition(p, rm_only=True, fix_cond=None):
    all_tz_conditions = list(p.env.tz.cond_dict.values())
    p_condition = p.env.tz.p_cond
    if fix_cond is not None:
        return fix_cond
    else:
        if rm_only:
            tz_cond = 'RM'
        else:
            tz_cond = np.random.choice(all_tz_conditions, p=p_condition)
        return tz_cond


def set_encoding_flag(t, enc_times, cond, agent):
    if t in enc_times and cond != 'NM':
        agent.encoding_on()
    else:
        agent.encoding_off()


def cond_manipulation(tz_cond, t, event_bond, hc_t, agent, n_lures=1):
    '''condition specific manipulation
    such as flushing, insert lure, etc.
    '''
    if t == event_bond:
        agent.retrieval_on()
        # flush WM unless RM
        if tz_cond != 'RM':
            hc_t = agent.get_init_states()
    return hc_t


# def cond_manipulation(tz_cond, t, event_bond, hc_t, agent, n_lures=1):
#     '''condition specific manipulation
#     such as flushing, insert lure, etc.
#     '''
#     if t == event_bond:
#         agent.retrieval_on()
#         if tz_cond == 'DM':
#             # RM: has EM, no WM
#             hc_t = agent.get_init_states()
#             agent.add_simple_lures(n_lures)
#         elif tz_cond == 'NM':
#             # RM: no WM, EM
#             hc_t = agent.get_init_states()
#             agent.flush_episodic_memory()
#             agent.add_simple_lures(n_lures+1)
#         elif tz_cond == 'RM':
#             # RM: has WM, EM
#             agent.add_simple_lures(n_lures)
#         else:
#             raise ValueError('unrecog tz condition')
#     return hc_t

# def append_prev_info(x_it_, a_prev, r_prev):
#     a_prev = a_prev.type(torch.FloatTensor).view(1)
#     r_prev = r_prev.type(torch.FloatTensor).view(1)
#     # y_prev = y_prev.type(torch.FloatTensor)
#     x_it = torch.cat([x_it_, a_prev, r_prev])
#     return x_it
