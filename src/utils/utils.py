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


def get_th_data(tensor_list):
    return [th_tensor.data for th_tensor in tensor_list]


def vprint(verbose, msg):
    if verbose:
        print(msg)
