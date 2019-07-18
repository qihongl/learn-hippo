import numpy as np
from utils.utils import to_sqnp
from utils.constants import TZ_COND_DICT
from analysis import compute_stats


def trim_data(n_examples_skip, data_list):
    return [data[n_examples_skip:] for data in data_list]


def compute_n_trials_to_skip(log_cond, p):
    # skip examples untill em buffer is full
    non_nm_trials = np.where(log_cond != TZ_COND_DICT.inverse['NM'])[0]
    n_examples_skip = non_nm_trials[p.n_event_remember+1]
    return n_examples_skip


def get_trial_cond_ids(log_cond):
    '''compute trial ids'''
    cond_ids = {}
    for cn in list(TZ_COND_DICT.values()):
        cond_id_ = TZ_COND_DICT.inverse[cn]
        cond_ids[cn] = log_cond == cond_id_
    return cond_ids


def process_cache(log_cache, T_total, p):
    # prealloc
    n_examples = len(log_cache)
    inpt = np.full((n_examples, T_total), np.nan)
    leak = np.full((n_examples, T_total), np.nan)
    comp = np.full((n_examples, T_total), np.nan)
    C = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
    H = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
    M = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
    CM = np.full((n_examples, T_total, p.net.n_hidden), np.nan)
    DA = np.full((n_examples, T_total, p.net.n_hidden_dec), np.nan)
    V = [None] * n_examples

    for i in range(n_examples):
        for t in range(T_total):
            # unpack data for i,t
            [vector_signal, scalar_signal, misc] = log_cache[i][t]
            [inpt_it, leak_it, comp_it] = scalar_signal
            [h_t, m_t, cm_t, des_act_t, V_i] = misc
            # cache data to np array
            inpt[i, t] = to_sqnp(inpt_it)
            leak[i, t] = to_sqnp(leak_it)
            comp[i, t] = to_sqnp(comp_it)
            H[i, t, :] = to_sqnp(h_t)
            M[i, t, :] = to_sqnp(m_t)
            CM[i, t, :] = to_sqnp(cm_t)
            DA[i, t, :] = to_sqnp(des_act_t)
            V[i] = V_i

    # compute cell state
    C = CM - M

    # pack data
    activity = [C, H, M, CM, DA, V]
    ctrl_param = [inpt, leak, comp]
    return activity, ctrl_param


'''data separator'''


def sep_by_qsource(matrix_p2, obj_uncertainty_info, n_se=3):
    [em_only_cond_p2, wm_only_cond_p2, neither_cond_p2,
     both_cond_p2] = obj_uncertainty_info
    n_param = np.shape(obj_uncertainty_info[0])[1]
    m_em_mu, m_em_er = np.zeros(n_param,), np.zeros(n_param,)
    m_wm_mu, m_wm_er = np.zeros(n_param,), np.zeros(n_param,)
    m_bo_mu, m_bo_er = np.zeros(n_param,), np.zeros(n_param,)
    m_nt_mu, m_nt_er = np.zeros(n_param,), np.zeros(n_param,)
    for t in range(n_param):
        m_em_mu[t], m_em_er[t] = compute_stats(
            matrix_p2[em_only_cond_p2[:, t], t], n_se=n_se)
        m_wm_mu[t], m_wm_er[t] = compute_stats(
            matrix_p2[wm_only_cond_p2[:, t], t], n_se=n_se)
        m_bo_mu[t], m_bo_er[t] = compute_stats(
            matrix_p2[both_cond_p2[:, t], t], n_se=n_se)
        m_nt_mu[t], m_nt_er[t] = compute_stats(
            matrix_p2[neither_cond_p2[:, t], t], n_se=n_se)
    stats = {
        'EM only': [m_em_mu, m_em_er],
        'neither': [m_nt_mu, m_nt_er],
        'both': [m_bo_mu, m_bo_er],
        'WM only': [m_wm_mu, m_wm_er],
    }
    return stats
