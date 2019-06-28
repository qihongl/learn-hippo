"""the episodic memory class
notes:
- there is no key - val distinction here
- memory is a row vector
"""
import torch
import torch.nn.functional as F
from models.utils import list2mat
from models.metrics import rbf, lca_transform


# constants
ALL_KERNELS = ['cosine', 'l1', 'l2', 'rbf']
ALL_POLICIES = ['LCA', '1NN', 'all']


class EM():
    """episodic memory

    Parameters
    ----------
    size : int
        the maximial len of the dictionary
    dim : int
        the dim or len of memory i, we assume memory_i is a row vector
    kernel : str
        the metric for memory search
    sigma : float
        the parameter for the gaussian kernel

    Attributes
    ----------
    encoding_off : bool
        if True, stop forming memories
    retrieval_off : type
        if True, stop retrieving memories
    reset_memory : func;
        if called, clear the dictionary
    _check_config : func
        check the class config

    size
    kernel
    dim
    sigma

    """

    def __init__(self, size, dim, kernel='cosine'):
        self.size = size
        self.dim = dim
        self.kernel = kernel
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
        self.vals.append(val.data.view(1, -1))
        # remove the oldest memory, if overflow
        if len(self.vals) > self.size:
            self.vals.pop(0)

    def save_memory(self, val):
        """Save an episodic memory

        Parameters
        ----------
        val : a row vector
            a EM value, representing the memory content
        """
        if self.encoding_off:
            return
        self._save_memory(val)

    def inject_memories(self, vals):
        """Inject pre-defined values

        Parameters
        ----------
        vals : list
            a list of memory content
        """
        for v in vals:
            self._save_memory(v)

    def get_memory(
            self, input_pattern,
            leak=None, comp=None, w_input=None
    ):
        """Perform a 1-models search over EM

        Parameters
        ----------
        input_pattern : a row vector
            a EM key, used to for memory search

        Returns
        -------
        a row vector
            a EM value, representing the memory content
        """
        # if no memory, return the zero vector
        if len(self.vals) == 0 or self.retrieval_off:
            return empty_memory(self.dim)
        return self._recall(
            input_pattern, leak=leak, comp=comp, w_input=w_input
        )

    # @torch.no_grad()
    def _recall(
            self, input_pattern,
            leak=None, comp=None, w_input=None
    ):
        """get the episodic memory according to some policy

        Parameters
        ----------
        input_pattern :

        Returns
        -------
        a row vector
            a EM value, representing the memory content
        """
        # compute similarity(query, memory_i ), for all i
        similarities = compute_similarities(
            input_pattern, self.vals, self.kernel
        )
        memory_wts = lca_transform(
            similarities, leak=leak, comp=comp, w_input=w_input
        ).view(1, -1)
        memory_matrix = list2mat(self.vals)
        retrieved_item = torch.mm(memory_wts, memory_matrix)
        return retrieved_item


"""helpers"""


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
    q = input_pattern.data.view(1, -1)
    # reshape memory keys to #keys x key_dim
    M = list2mat(vals)
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
def empty_memory(dim):
    return torch.zeros(1, dim).data
