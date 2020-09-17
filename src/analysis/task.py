import numpy as np
from itertools import product
from analysis.utils import one_hot_to_int


def get_oq_keys(X_i, task, to_int=True):
    """extract obs/query keys from the input matrix, for one sample

    Parameters
    ----------
    X_i : np array
        a sample from SequenceLearning task
    task : object
        the SequenceLearning task that generated X_i
    to_int : bool
        whether convert to integer representation

    Returns
    -------
    list, list, list
        observation keys, query keys, observation values

    """
    # get the observation / query keys
    o_key = X_i[:, :task.k_dim]
    q_key = X_i[:, -task.k_dim:]
    o_val = X_i[:, task.k_dim:task.k_dim + task.v_dim]
    # convert to integer representation
    if to_int:
        o_key = [one_hot_to_int(o_key[t]) for t in range(len(o_key))]
        q_key = [one_hot_to_int(q_key[t]) for t in range(len(q_key))]
        o_val = [one_hot_to_int(o_val[t]) for t in range(len(o_val))]
    return o_key, q_key, o_val


def set_nanadd(input_set, new_element):
    """set.add a new element, don't add np.nan

    Parameters
    ----------
    input_set : set
        a set of int
    new_element : int
        a new element to be added to the set

    Returns
    -------
    set
        the set updated by the new element

    """
    if not np.isnan(new_element):
        input_set.add(new_element)
    return input_set


def _compute_true_dk(o_key, q_key, o_val, task):
    """compute ground truth uncertainty for a trial

    Parameters
    ----------
    o_key : list of int
        Description of parameter `o_key`.
    q_key : list of int
        Description of parameter `q_key`.
    o_val : list of int
        Description of parameter `o_val`.
    task : obj
        the SL task

    Returns
    -------
    type
        Description of returned object.

    """
    assert task.n_parts == 2, 'this function only works for 2-part seq'
    assert len(o_key) == len(q_key), 'obs seq length must match query seq'
    T_total_ = len(o_key)
    # T_part_ = T_total_ // task.n_parts
    # prealloc
    o_key_up_to_t, q_key_up_to_t = set(), set()
    dk = np.ones(T_total_, dtype=bool)
    # compute uncertainty info over time
    for t in range(T_total_):
        q_key_up_to_t = set_nanadd(q_key_up_to_t, q_key[t])
        # if the observation is not nan (removed), consider it as an observed key
        if not np.isnan(o_val[t]):
            # if the key is not nan (due to delay), add it as an observed key
            o_key_up_to_t = set_nanadd(o_key_up_to_t, o_key[t])
        # if the query is in the observed key up to time t
        if q_key[t] in o_key_up_to_t:
            # shouldn't say don't know
            dk[t] = False
        # log info
        # t_relative = np.mod(t, T_part_)
        # print(f'time = {t}, {t_relative} / {T_total_} |  dk = {dk[t]}')
        # print(o_key_up_to_t)
        # print(q_key_up_to_t)
    return dk


def compute_true_dk(X_i, task):
    """compute objective uncertainty w/ or w/o EM (EM vs. WM), where ...
    - with EM == no flusing, which applies to the RM condition
    - WM == w/o EM == EM flushed, which applies to the NM and DM

    Parameters
    ----------
    X_i : np array
        a sample from SequenceLearning task
    task : object
        the SequenceLearning task that generated X_i

    Returns
    -------
    dict
        ground truth / objective uncertainty

    """
    assert task.n_parts == 2, 'this function only works for 2-part seq'
    o_key, q_key, o_val = get_oq_keys(X_i, task, to_int=True)
    T_total_ = len(o_key)
    T_part_ = T_total_ // task.n_parts
    dk = {}
    dk['EM'] = _compute_true_dk(o_key, q_key, o_val, task)
    dk['WM'] = _compute_true_dk(
        o_key[T_part_:], q_key[T_part_:], o_val[T_part_:], task
    )
    return dk


def batch_compute_true_dk(X, task, dtype=bool):
    """compute the uncertainty ground truth for a sample/batch of data
    - a wrapper for `compute_true_dk()`

    Parameters
    ----------
    X : 3d array
        a sample from the SL task
    task : obj
        the SL task

    Returns
    -------
    2d array, 2d array
        uncertainty w/ w/o episodic flush

    """
    n_samples = len(X)
    dk_wm = np.zeros((n_samples, task.n_param), dtype=dtype)
    dk_em = np.zeros((n_samples, task.n_param * task.n_parts), dtype=dtype)
    # dk = [compute_true_dk(X[i], task) for i in range(n_samples)]
    # pred_time_mask = [None] * n_samples
    for i in range(n_samples):
        T_total_i = np.shape(X[i])[0]
        T_part_i, pad_len_i, _, _ = task.get_time_param(T_total_i)
        pred_time_mask_i = task.get_pred_time_mask(
            T_total_i, T_part_i, pad_len_i)
        # compute objective uncertainty, w/ or w/o EM
        dk_i = compute_true_dk(X[i], task)
        dk_wm[i] = dk_i['WM'][pred_time_mask_i[T_part_i:]]
        dk_em[i] = dk_i['EM'][pred_time_mask_i]
    return dk_wm, dk_em


def compute_event_similarity_matrix(Y, normalize=False):
    """compute the inter-event similarity matrix of a batch of data

    e.g.
    task = SequenceLearning(n_param, n_branch, n_parts=1)
    X, Y = task.sample(n_samples)
    similarity_matrix = compute_event_similarity_matrix(Y, normalize=False)

    Parameters
    ----------
    Y : 3d array (n_examples, _, _) or 2d array (n_examples, _)
        the target values
    normalize : bool
        whether to normalize by vector dim

    Returns
    -------
    2d array (n_examples, n_examples)
        the inter-event similarity matrix

    """
    if len(np.shape(Y)) == 3:
        Y_int = np.argmax(Y, axis=-1)
    elif len(np.shape(Y)) == 2:
        Y_int = Y
    else:
        raise ValueError('Invalid Y shape')
    # prealloc
    n_samples = np.shape(Y)[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i, j in product(range(n_samples), range(n_samples)):
        similarity_matrix[i, j] = compute_event_similarity(
            Y_int[i], Y_int[j], normalize=normalize)
    return similarity_matrix


def compute_event_similarity(event_i, event_j, normalize=True):
    """compute the #shared elements for two arrays
    e.g.
    event_i = np.argmax(q_vals_vec[i], axis=-1)
    event_j = np.argmax(q_vals_vec[j], axis=-1)
    sim_ij = compute_event_similarity(event_i, event_j, normalize=True)

    Parameters
    ----------
    event_i/j : 1d np array
        event representation
    normalize : bool
        whether to normalize by vector dim

    Returns
    -------
    float
        similarity

    """
    assert np.shape(event_i) == np.shape(event_j)
    similarity = np.sum(event_i == event_j)
    if normalize:
        return similarity / len(event_i)
    return similarity

#
# def remove_identical_events(Ys, n_param):
#     """remove events that are identical
#
#     Parameters
#     ----------
#     Ys : 3d array
#         number of examples x number of time points x feature dim
#     n_param : int
#         indicate max(number of shared parameters)
#
#     Returns
#     -------
#     Ys : 3d array
#         number of examples' x number of time points x feature dim
#
#     """
#     event_similarity_matrix = compute_event_similarity(Ys, tril_k=-1)
#     repeated_id = np.where(event_similarity_matrix == n_param)
#     rm_axis = 0
#     Ys_ = np.delete(Ys, repeated_id[rm_axis], axis=rm_axis)
#     return Ys_
