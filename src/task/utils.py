import numpy as np


def get_event_ends(T_part, n_repeats):
    """get the end points for a event sequence, with lenth T, and k repeats
    - event ends need to be removed for prediction accuracy calculation, since
    there is nothing to predict there
    - event boundaries are defined by these values

    Parameters
    ----------
    T_part : int
        the length of an event sequence (one repeat)
    n_repeats : int
        number of repeats

    Returns
    -------
    1d np.array
        the end points of event seqs

    """
    return [T_part * (k+1)-1 for k in range(n_repeats)]


def sample_def_tps(n_param, n_def_tps):
    def_tps = np.zeros(n_param,)
    if n_def_tps == 0:
        return list(def_tps)
    def_tps_ids = np.random.choice(np.arange(n_param), n_def_tps, replace=False)
    def_tps[def_tps_ids] = 1
    return list(def_tps)


def sample_rand_path(B, T):
    """sample a random path on the event graph

    Parameters
    ----------
    B : int
        branching factor
    T : int
        the length of the event / path

    Returns
    -------
    type
        T x B matrix, each row is a B-dimensional one hot vector

    """
    return np.array([get_random_one_hot_vector(B) for t in range(T)])


def get_random_one_hot_vector(k, probs=None):
    """get a random one hot vector of dimension k
    - the one hot POSITION is random

    Parameters
    ----------
    k : int
        the dim of the identity
    probs : list of float / 1d array
        the distribution over locations

    Returns
    -------
    1d array
        a one-hot vector
    """
    loc_id = np.random.choice(k, p=probs)
    return get_one_hot_vector(loc_id, k)


def get_one_hot_vector(i, k):
    """get the i-th k-dim one hot vector

    Parameters
    ----------
    i : type
        Description of parameter `i`.
    k : type
        Description of parameter `k`.

    Returns
    -------
    type
        Description of returned object.

    """
    return np.eye(k)[i, :]


def chunks(lst, chunk_size):
    '''
    Adapted from:
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    '''
    assert chunk_size >= 1
    return [lst[i:i+chunk_size] for i in np.arange(0, len(lst), chunk_size)]


def get_scrambled_ids(n_time_points, nchunks):
    assert n_time_points % nchunks == 0
    chunk_size = n_time_points // nchunks
    tp_chunks = chunks(np.arange(n_time_points), chunk_size)
    tp_chunks_scrambled = tp_chunks[::-1]
    tp_scrambled = np.concatenate(tp_chunks_scrambled)
    return tp_scrambled


def scramble_array(nparray, nchunks=4):
    '''scramble the input array along the 0-th axis

    e.g.
    X = np.random.normal(size=(T, V))
    X_scrb = scramble_array(X, nchunks=4)
    '''
    n_tps = np.shape(nparray)[0]
    tp_scrambled = get_scrambled_ids(n_tps, nchunks)
    return nparray[tp_scrambled, :]


def scramble_array_list(array_list, nchunks=4):
    '''scramble a list of input arrays along the 0-th axis
    '''
    return [scramble_array(array) for array in array_list]


# def get_ith_nCk(index, n, k):
#     """get the i-th return from the set of n-choose-k
#
#     Yields the items of the single combination that would be at the provided
#     (0-based) index in a lexicographically sorted list of combinations of choices
#     of k items from n items [0,n), given the combinations were sorted in
#     descending order. Yields in descending order.
#     https://stackoverflow.com/questions/1776442/nth-combination
#
#     Parameters
#     ----------
#     index : type
#         Description of parameter `index`.
#     n : type
#         Description of parameter `n`.
#     k : type
#         Description of parameter `k`.
#
#     Returns
#     -------
#     type
#         Description of returned object.
#
#     """
#     nCk = 1
#     for nMinusI, iPlus1 in zip(range(n, n - k, -1), range(1, k + 1)):
#         nCk *= nMinusI
#         nCk //= iPlus1
#     curIndex = nCk
#     for k in range(k, 0, -1):
#         nCk *= k
#         nCk //= n
#         while curIndex - nCk > index:
#             curIndex -= nCk
#             nCk *= (n - k)
#             nCk -= nCk % k
#             n -= 1
#             nCk //= n
#         n -= 1
#         yield n


def get_botvinick_query(n_param, n_events=1, n_parts=2):
    """get botvinick stimuli - repeat query for 1st vs. 2nd half of the event

    Parameters
    ----------
    n_param : int
        Description of parameter `n_param`.
    n_events : int
        Description of parameter `n_events`.
    n_parts : int
        Description of parameter `n_parts`.

    Returns
    -------
    type
        Description of returned object.

    """

    return np.hstack([
        np.vstack([np.eye(n_param // 2) for _ in range(n_parts * n_events)]),
        np.vstack([np.zeros((n_param // 2, n_param // 2))
        for _ in range(n_parts * n_events)])
    ])


if __name__ == "__main__":
    '''show how to use get_botvinick_query'''
    import matplotlib.pyplot as plt
    n_param = 16
    n_parts = 2
    bquery = get_botvinick_query(n_param, n_events=1)
    plt.imshow(bquery)
