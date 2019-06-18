import torch


'''data formatting'''


def list2mat(list_of_vectors):
    """convert a list of ROW vectors to a torch matrix

    Parameters
    ----------
    list_of_vectors : list
        a list of ROW vectors

    Returns
    -------
    a torch matrix
        Description of returned object.

    """
    n_vectors = len(list_of_vectors)
    mat = torch.stack(list_of_vectors, dim=1).view(n_vectors, -1)
    return mat
