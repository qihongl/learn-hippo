import numpy as np
from utils.utils import to_sqnp
from utils.constants import TZ_COND_DICT
from analysis import compute_stats


def remove_none(input_list, return_missing_idx=False):
    updated_list = []
    missing_ids = []
    for i, item in enumerate(input_list):
        if item is not None:
            updated_list.append(item)
        else:
            missing_ids.append(i)
    if return_missing_idx:
        return updated_list, np.array(missing_ids)
    return updated_list


def trim_data(n_examples_skip, data_list):
    return [data[n_examples_skip:] for data in data_list]


def compute_n_trials_to_skip(log_cond, p):
    # skip examples untill em buffer is full
    non_nm_trials = np.where(log_cond != TZ_COND_DICT.inverse['NM'])[0]
    n_examples_skip = non_nm_trials[p.n_event_remember + 1]
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
            # [inpt_it] = scalar_signal
            # [inpt_it, _, _] = scalar_signal
            [inpt_it, leak_it, comp_it] = scalar_signal
            [h_t, m_t, cm_t, des_act_t, V_i] = misc
            # cache data to np array
            # leak[i, t] = leak_it
            # comp[i, t] = to_sqnp(comp_it)
            inpt[i, t] = to_sqnp(inpt_it)
            H[i, t, :] = to_sqnp(h_t)
            M[i, t, :] = to_sqnp(m_t)
            CM[i, t, :] = to_sqnp(cm_t)
            DA[i, t, :] = to_sqnp(des_act_t)
            V[i] = V_i
    # compute cell state
    C = CM - M
    # pack data
    activity = [C, H, M, CM, DA, V]
    ctrl_param = [inpt]
    # ctrl_param = [inpt, leak, comp]
    return activity, ctrl_param


'''data separator'''


def get_qsource(true_dk_em, true_dk_wm, cond_ids, p):
    """compute query source

    Parameters
    ----------
    true_dk_em : type
        Description of parameter `true_dk_em`.
    true_dk_wm : type
        Description of parameter `true_dk_wm`.
    cond_ids : type
        Description of parameter `cond_ids`.
    p : type
        Description of parameter `p`.

    Returns
    -------
    type
        Description of returned object.

    """
    # DM
    # can recall from EM
    true_dk_em_dm_p2 = true_dk_em[cond_ids['DM'], p.env.n_param:]
    true_dk_wm_dm_p2 = true_dk_wm[cond_ids['DM'], :]
    eo_dm_p2 = np.logical_and(true_dk_wm_dm_p2, ~true_dk_em_dm_p2)
    wo_dm_p2 = np.logical_and(~true_dk_wm_dm_p2, true_dk_em_dm_p2)
    nt_dm_p2 = np.logical_and(true_dk_wm_dm_p2, true_dk_em_dm_p2)
    bt_dm_p2 = np.logical_and(~true_dk_wm_dm_p2, ~true_dk_em_dm_p2)
    # NM
    # no episodic memory => "EM only", "both" are impossble
    # true_dk_em_nm_p2 = true_dk_em[cond_ids['NM'], n_param:]
    true_dk_wm_nm_p2 = true_dk_wm[cond_ids['NM'], :]
    n_trials_, n_time_steps_ = np.shape(true_dk_wm_nm_p2)
    eo_nm_p2 = np.zeros(shape=(n_trials_, n_time_steps_), dtype=bool)
    wo_nm_p2 = ~true_dk_wm_nm_p2
    nt_nm_p2 = true_dk_wm_nm_p2
    bt_nm_p2 = np.zeros(shape=(n_trials_, n_time_steps_), dtype=bool)
    # RM
    # has episodic memory
    true_dk_rm_nm_p2 = true_dk_em[cond_ids['RM'], p.env.n_param:]
    n_trials_, n_time_steps_ = np.shape(true_dk_wm_nm_p2)
    eo_rm_p2 = np.zeros(shape=(n_trials_, n_time_steps_), dtype=bool)
    wo_rm_p2 = np.zeros(shape=(n_trials_, n_time_steps_), dtype=bool)
    nt_rm_p2 = true_dk_rm_nm_p2
    bt_rm_p2 = ~true_dk_rm_nm_p2
    # gather data
    q_source_rm = {
        'EM only': eo_rm_p2, 'WM only': wo_rm_p2,
        'neither': nt_rm_p2, 'both': bt_rm_p2
    }
    q_source_dm = {
        'EM only': eo_dm_p2, 'WM only': wo_dm_p2,
        'neither': nt_dm_p2, 'both': bt_dm_p2
    }
    q_source_nm = {
        'EM only': eo_nm_p2, 'WM only': wo_nm_p2,
        'neither': nt_nm_p2, 'both': bt_nm_p2
    }
    #
    q_source_all_conds = {
        'RM': q_source_rm,
        'DM': q_source_dm,
        'NM': q_source_nm
    }
    return q_source_all_conds


def sep_by_qsource(matrix_p2, q_source_info, n_se=3):
    """separate values by query source and then compute statistics

    Parameters
    ----------
    matrix_p2 : type
        Description of parameter `matrix_p2`.
    q_source_info : type
        Description of parameter `q_source_info`.
    n_se : type
        Description of parameter `n_se`.

    Returns
    -------
    type
        Description of returned object.

    """
    stats = {}
    # loop over sources
    for q_source_name, q_source_id in q_source_info.items():
        T_ = np.shape(q_source_id)[1]
        mu_, er_ = np.zeros(T_,), np.zeros(T_,)
        # loop over time
        for t in range(T_):
            # compute stats
            mu_[t], er_[t] = compute_stats(
                matrix_p2[q_source_id[:, t], t], n_se=n_se)
        # collect stats for this source
        stats[q_source_name] = [mu_, er_]
    return stats
