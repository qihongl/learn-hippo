"""the episodic memory class
notes:
- there is no key - val distinction here
- memory is a row vector
"""
import torch
import torch.nn.functional as F
from copy import deepcopy
from models.LCA_pytorch import LCA

# constants
ALL_KERNELS = ['cosine', 'l1', 'l2', 'rbf']


class EM():
    """An episodic memory memchanism

    Parameters
    ----------
    size : int
        the storage capacity
    dim : int
        the dim or len of an episodic memory i
    kernel : str
        the metric for memory-cell_state similarity evaluation

    """

    def __init__(self, size, dim, kernel='cosine'):
        self.size = size
        self.dim = dim
        self.kernel = kernel
        self.vals = []
        self.reset_memory()
        self._check_config()

    def reset_memory(self):
        self.flush()
        self.encoding_off = True
        self.retrieval_off = True

    def _check_config(self):
        assert self.size > 0
        assert self.kernel in ALL_KERNELS

    def flush(self):
        self.vals = []

    def _save_memory(self, val):
        self.vals.append(torch.squeeze(val.data))
        # remove the oldest memory, if overflow
        if len(self.vals) > self.size:
            self.vals.pop(0)

    def remove_memory(self, id):
        assert id <= len(self.vals) - 1, 'index out of bound'
        return self.vals.pop(id)

    def save_memory(self, val):
        """Save an episodic memory

        Parameters
        ----------
        val : torch.tensor
            a memory
        """
        if self.encoding_off:
            return
        self._save_memory(val)

    def inject_memories(self, vals):
        """Inject pre-defined values

        Parameters
        ----------
        vals : list
            a list of memories
        """
        for v in vals:
            self._save_memory(v)

    def get_memory(
            self, input_pattern,
            leak=None, comp=None, w_input=None
    ):
        """given a cortical pattern, return an episodic memory

        Parameters
        ----------
        input_pattern : torch.tensor
            a cortical pattern, served as a key for memory search

        Returns
        -------
        torch.tensor
            a memory
        """
        # if no memory, return the zero vector
        if len(self.vals) == 0 or self.retrieval_off:
            return dummy_memory(self.dim)
        return self._get_memory(
            input_pattern, leak=leak, comp=comp, w_input=w_input
        )

    # @torch.no_grad()
    def _get_memory(
            self, input_pattern,
            leak=None, comp=None, w_input=None
    ):
        """get an episodic memory

        Parameters
        ----------
        input_pattern : torch.tensor
            a cortical pattern, served as a key for memory search

        Returns
        -------
        torch.tensor
            a memory
        """
        # compute similarity(query, memory_i ), for all i
        w_raw = compute_similarities(
            input_pattern, self.vals, self.kernel
        )
        w = lca_transform(
            w_raw, leak=leak, comp=comp, w_input=w_input
        ).view(1, -1)
        # compute the memory matrix
        M = torch.stack(self.vals)
        return w @ M

    def get_vals(self):
        return deepcopy(self.vals)


"""helpers"""


def transform_similarities(
        raw_similarities, weighting_function,
        leak=None, comp=None, w_input=None
):
    n_memories = len(raw_similarities)
    if weighting_function == '1NN':
        # one hot vector weighting for 1NN
        best_memory_id = torch.argmax(raw_similarities)
        similarities = torch.eye(n_memories)[best_memory_id]
    elif weighting_function == 'LCA':
        # transform the similarity by a LCA process
        similarities = lca_transform(
            raw_similarities,
            leak=leak, comp=comp, w_input=w_input
        )
    else:
        similarities = raw_similarities
    return similarities


def compute_similarities(
        input_pattern, vals, metric,
):
    """Compute the similarity between query vs. key_i for all i
        i.e. compute q M, w/ q: 1 x key_dim, M: #keys x key_dim

    Parameters
    ----------
    input_pattern : a row vector
        Description of parameter `input_pattern`.
    vals : list
        Description of parameter `vals`.
    metric : str
        Description of parameter `metric`.

    Returns
    -------
    a row vector w/ len #memories
        the similarity between query vs. key_i, for all i
    """
    # reshape query to 1 x key_dim
    q = input_pattern.view(1, -1)
    # reshape memory keys to #keys x key_dim
    M = torch.stack(vals)
    # compute similarities
    if metric == 'cosine':
        similarities = F.cosine_similarity(q, M)
    elif metric == 'l1':
        similarities = - F.pairwise_distance(q, M, p=1)
    elif metric == 'l2':
        similarities = - F.pairwise_distance(q, M, p=2)
    else:
        raise Exception(f'Unrecognizable metric: {metric}')
    return similarities


@torch.no_grad()
def dummy_memory(dim):
    return torch.zeros(1, dim).data


def lca_transform(
        similarities,
        leak=None, comp=None, w_input=None, n_cycles=10
):
    # construct input sequence
    stimuli = similarities.repeat(n_cycles, 1)
    # init LCA
    lca = LCA(
        n_units=len(similarities),
        leak=leak, ltrl_inhib=comp, w_input=w_input,
    )
    # run LCA
    lca_outputs = lca.run(stimuli)
    # take the final valuse
    lca_similarities = lca_outputs[-1, :]
    return lca_similarities


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy
    import numpy as np
    import torch

    '''how to init'''
    size, dim = 4, 8
    em = EM(size, dim)
    em.encoding_off = False

    '''how to add memory'''
    n_mem = 2
    for i in range(n_mem):
        m = torch.ones(size=(dim,)) * i
        em.save_memory(m)
    print(em.vals)

    '''how to delete a specific memory'''
    # make a copy 1st
    em_vals = deepcopy(em.vals)
    # remove the 2nd to last memory
    mem_rmd = em.remove_memory(-2)
    print('before removal:')
    print(em_vals)
    print('after:')
    print(em.vals)
    print('the removed memory:')
    print(mem_rmd)
    # recover the memories by the deepcopy
    em.vals = em_vals
    print(em.vals)
