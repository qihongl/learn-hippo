import numpy as np


def compute_event_similarity(event_i, event_j, normalize=True):
    """compute the #shared elements for two arrays

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


# def compute_event_similarity(Ys, tril_k=None):
#     """compute event parameter overlap matrix - i.e. the similarity matrix
#
#     e.g.
#     n_param, n_branch = 6, 3
#     n_movies, n_parts = 100, 1
#     def_prob = .6
#     mg = MovieGen(n_param, n_branch, def_prob=def_prob)
#     Xs, Ys, param_removed = mg.gen_movies(n_movies, n_parts, stack=True)
#     S = compute_event_similarity(Ys, tril=False)
#
#     Parameters
#     ----------
#     Ys : 3d array
#         number of examples x number of time points x feature dim
#     tril : bool
#         return lower triangular part only
#
#     Returns
#     -------
#     2d array
#         inter-event similarity matrix
#
#     """
#     assert len(np.shape(Ys)) == 3
#     Y_feature_vectors = np.sum(Ys, axis=1)
#     event_similarity_matrix = Y_feature_vectors @ Y_feature_vectors.T
#     if tril_k is not None:
#         event_similarity_matrix = np.tril(event_similarity_matrix, k=tril_k)
#     return event_similarity_matrix
#
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
