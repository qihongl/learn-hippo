import torch
import numpy as np


def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.tensor(np_array).type(pth_dtype)


def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))


def to_np(torch_tensor):
    return torch_tensor.data.numpy()


def to_sqnp(torch_tensor):
    return np.squeeze(to_np(torch_tensor))


def batch_sqnp(list_of_tensors):
    return [to_sqnp(tsr) for tsr in list_of_tensors]


def get_th_data(tensor_list):
    return [th_tensor.data for th_tensor in tensor_list]


def vprint(verbose, msg):
    if verbose:
        print(msg)


def find_factors(x):
    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            factors.append(i)
    return factors


def chunk(lst, n_chunks):
    """
    https://stackoverflow.com/questions/2130016/
    splitting-a-list-into-n-parts-of-approximately-equal-length

    Parameters
    ----------
    lst : list
    n_chunks : int

    Returns
    -------
    list
        chunked list

    """
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(n_chunks)]
