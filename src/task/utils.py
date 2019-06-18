import numpy as np

'''simple helpers'''


def param_id_to_one_hot_dims(pid, n_branch):
    """map parameter id to the relevant dimension over one hot rep

    Parameters
    ----------
    pid : int
        parameter id
    n_branch : int
        branching factor of the graph

    Returns
    -------
    list
        the relevant dimensions in the one-hot representation

    """
    return np.arange(n_branch) + pid * n_branch


def permute_rows(matrix):
    """temporally permute the order of events

    Parameters
    ----------
    matrix : 2d array
        typically the input observation matrix

    Returns
    -------
    2d array
        the row-permuted matix

    """
    n_rows, _ = np.shape(matrix)
    perm_op = np.random.permutation(range(n_rows))
    return matrix[perm_op, :]


def sample_rand_path(B, T):
    """sample a random path

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
    return np.eye(k)[loc_id, :]


def compute_event_similarity(Ys, tril_k=None):
    """compute event parameter overlap matrix - i.e. the similarity matrix

    e.g.
    n_param, n_branch = 6, 3
    n_movies, n_parts = 100, 1
    def_prob = .6
    mg = MovieGen(n_param, n_branch, def_prob=def_prob)
    Xs, Ys, param_removed = mg.gen_movies(n_movies, n_parts, stack=True)
    S = compute_event_similarity(Ys, tril=False)

    Parameters
    ----------
    Ys : 3d array
        number of examples x number of time points x feature dim
    tril : bool
        return lower triangular part only

    Returns
    -------
    2d array
        inter-event similarity matrix

    """
    assert len(np.shape(Ys)) == 3
    Y_feature_vectors = np.sum(Ys, axis=1)
    event_similarity_matrix = Y_feature_vectors @ Y_feature_vectors.T
    if tril_k is not None:
        event_similarity_matrix = np.tril(event_similarity_matrix, k=tril_k)
    return event_similarity_matrix


def remove_identical_events(Ys, n_param):
    """remove events that are identical

    Parameters
    ----------
    Ys : 3d array
        number of examples x number of time points x feature dim
    n_param : int
        indicate max(number of shared parameters)

    Returns
    -------
    Ys : 3d array
        number of examples' x number of time points x feature dim

    """
    event_similarity_matrix = compute_event_similarity(Ys, tril_k=-1)
    repeated_id = np.where(event_similarity_matrix == n_param)
    rm_axis = 0
    Ys_ = np.delete(Ys, repeated_id[rm_axis], axis=rm_axis)
    return Ys_

#
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
