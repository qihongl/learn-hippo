import torch
import numpy as np
import torch.nn.functional as F
import pdb

from analysis import entropy
from utils.utils import to_sqnp
from utils.constants import TZ_COND_DICT, P_TZ_CONDS
from models import get_reward, compute_returns, compute_a2c_loss
# from task.utils import scramble_array, scramble_array_list


def run_aba(
        agent, optimizer, task, p, n_examples, supervised,
        fix_cond=None, fix_penalty=None,
        slience_recall_time=None, scramble=False,
        learning=True, get_cache=True, get_data=False,
):
    # sample data
    X, Y = task.sample(n_examples, interleave=True, to_torch=True)
    n_examples = n_examples // 2
    # logger
    log_return, log_pi_ent = 0, 0
    log_loss_sup, log_loss_actor, log_loss_critic = 0, 0, 0
    log_cond = np.zeros(n_examples,)
    log_dist_a = [[] for _ in range(n_examples)]
    log_targ_a = [[] for _ in range(n_examples)]
    log_cache = [None] * n_examples

    for i in range(n_examples):
        # pick a condition
        cond_i = pick_condition(p, rm_only=supervised, fix_cond=fix_cond)
        # get the example for this trial
        X_i, Y_i = X[i], Y[i]
        T_total = np.shape(X_i)[0]
        event_ends = np.array(
            [t for t in range(T_total+1) if t % p.env.n_param == 0][1:]
        ) - 1

        # prealloc
        loss_sup = 0
        probs, rewards, values, ents = [], [], [], []
        log_cache_i = [None] * T_total

        # init model wm and em
        penalty_val, penalty_rep = sample_penalty(p, fix_penalty)

        hc_t = agent.get_init_states()
        agent.flush_episodic_memory()
        agent.retrieval_on()
        agent.encoding_off()

        for t in range(T_total):
            if t in event_ends and cond_i != 'NM':
                agent.encoding_on()
            else:
                agent.encoding_off()

            # forward
            x_it = append_prev_info(X_i[t], [penalty_rep])
            pi_a_t, v_t, hc_t, cache_t = agent.forward(
                x_it.view(1, 1, -1), hc_t)
            # after delay period, compute loss
            a_t, p_a_t = agent.pick_action(pi_a_t)
            # get reward
            r_t = get_reward(a_t, Y_i[t], penalty_val)

            # cache the results for later RL loss computation
            rewards.append(r_t)
            values.append(v_t)
            probs.append(p_a_t)
            ents.append(entropy(pi_a_t))
            # compute supervised loss
            yhat_t = torch.squeeze(pi_a_t)[:-1]
            loss_sup += F.mse_loss(yhat_t, Y_i[t])

            # flush at event boundary
            if t in event_ends and cond_i != 'RM':
                hc_t = agent.get_init_states()

            # cache results for later analysis
            if get_cache:
                log_cache_i[t] = cache_t
            # for behavioral stuff, only record prediction time steps
            log_dist_a[i].append(to_sqnp(pi_a_t))
            log_targ_a[i].append(to_sqnp(Y_i[t]))

        # compute RL loss
        returns = compute_returns(rewards, normalize=p.env.normalize_return)
        loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
        pi_ent = torch.stack(ents).sum()

        if learning:
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
        log_cond[i] = TZ_COND_DICT.inverse[cond_i]
        if get_cache:
            log_cache[i] = log_cache_i

    # return cache
    log_dist_a = np.array(log_dist_a)
    log_targ_a = np.array(log_targ_a)
    results = [log_dist_a, log_targ_a, log_cache, log_cond]
    metrics = [log_loss_sup, log_loss_actor, log_loss_critic,
               log_return, log_pi_ent]
    out = [results, metrics]
    if get_data:
        X_array_list = [to_sqnp(X[i]) for i in range(n_examples)]
        Y_array_list = [to_sqnp(Y[i]) for i in range(n_examples)]
        training_data = [X_array_list, Y_array_list]
        out.append(training_data)
    return out


def append_prev_info(x_it_, scalar_list):
    for s in scalar_list:
        x_it_ = torch.cat(
            [x_it_, s.type(torch.FloatTensor).view(tensor_length(s))]
        )
    return x_it_


def tensor_length(tensor):
    if tensor.dim() == 0:
        length = 1
    elif tensor.dim() > 1:
        raise ValueError('length for high dim tensor is undefined')
    else:
        length = len(tensor)
    return length


def pick_condition(p, rm_only=True, fix_cond=None):
    all_tz_conditions = list(TZ_COND_DICT.values())
    if fix_cond is not None:
        return fix_cond
    else:
        if rm_only:
            tz_cond = 'RM'
        else:
            tz_cond = np.random.choice(all_tz_conditions, p=P_TZ_CONDS)
        return tz_cond


def sample_penalty(p, fix_penalty):
    # if penalty level is fixed, usually used during test
    if fix_penalty is not None:
        penalty_val = fix_penalty
    else:
        # otherwise sample a penalty level
        if p.env.penalty_random:
            if p.env.penalty_discrete:
                penalty_val = np.random.choice(p.env.penalty_range)
            else:
                penalty_val = np.random.uniform(0, p.env.penalty)
        else:
            # or train with a fixed penalty level
            penalty_val = p.env.penalty
    # form the input representation of the current penalty signal
    if p.env.penalty_onehot:
        penalty_rep = one_hot_penalty(penalty_val, p)
    else:
        penalty_rep = penalty_val
    return torch.tensor(penalty_val), torch.tensor(penalty_rep)


def one_hot_penalty(penalty_int, p):
    assert penalty_int in p.env.penalty_range, \
        print(f'invalid penalty_int = {penalty_int}')
    one_hot_dim = len(p.env.penalty_range)
    penalty_id = p.env.penalty_range.index(penalty_int)
    return np.eye(one_hot_dim)[penalty_id, :]
