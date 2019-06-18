"""the DND class
notes:
- memory is a row vector
"""
import torch
import torch.nn.functional as F
from models.utils import list2mat
from models.metrics import rbf, lca_transform


# constants
ALL_KERNELS = ['cosine', 'l1', 'l2', 'rbf']
ALL_POLICIES = ['LCA', '1NN', 'all']


class DND():
    """The differentiable neural dictionary (DND) class. This enables episodic
    recall in a neural network.

    Parameters
    ----------
    dict_len : int
        the maximial len of the dictionary
    memory_dim : int
        the dim or len of memory i, we assume memory_i is a row vector
    kernel : str
        the metric for memory search
    recall_func : str
        1NN: get the best memory
        thres1NN: get the best memory if distance-to-best < delta
        all: retrieve a weighted combination of all memories
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

    dict_len
    kernel
    memory_dim
    recall_func
    sigma

    """

    def __init__(
            self, dict_len, memory_dim, kernel, recall_func,
    ):
        self.dict_len = dict_len
        self.memory_dim = memory_dim
        self.kernel = kernel
        self.recall_func = recall_func
        self.reset_memory()

    def reset_memory(self):
        self.flush()
        # turn off recall/encoding by default
        self.encoding_off = True
        self.retrieval_off = True
        #
        self._check_config()

    def flush(self):
        # empty the memories
        self.keys = []
        self.vals = []

    def _check_config(self):
        assert self.dict_len > 0
        assert self.kernel in ALL_KERNELS
        assert self.recall_func in ALL_POLICIES

    def inject_memories(self, keys, vals):
        """Inject pre-defined keys and values

        Parameters
        ----------
        keys : list
            a list of memory keys
        vals : list
            a list of memory content
        """
        assert len(keys) == len(vals)
        for k, v in zip(keys, vals):
            self._save_memory(k, v)
            # remove the oldest memory, if overflow
            if len(self.keys) > self.dict_len:
                self.keys.pop(0)
                self.vals.pop(0)

    def _save_memory(self, key, val):
        # get data is necessary for gradient reason
        self.keys.append(key.data.view(1, -1))
        self.vals.append(val.data.view(1, -1))

    def save_memory(self, key, val):
        """Save an episodic memory to the dictionary

        Parameters
        ----------
        key : a row vector
            a DND key, used to for memory search
        val : a row vector
            a DND value, representing the memory content
        """
        if self.encoding_off:
            return
        # add new memory to the the dictionary
        self._save_memory(key, val)
        # remove the oldest memory, if overflow
        if len(self.keys) > self.dict_len:
            self.keys.pop(0)
            self.vals.pop(0)

    def get_memory(
            self, query_key,
            sigma=None, leak=None, comp=None, w_input=None
    ):
        """Perform a 1-models search over dnd

        Parameters
        ----------
        query_key : a row vector
            a DND key, used to for memory search

        Returns
        -------
        a row vector
            a DND value, representing the memory content
        """
        # if no memory, return the zero vector
        if len(self.keys) == 0 or self.retrieval_off:
            return empty_memory(self.memory_dim)
        return self._recall(
            query_key, sigma=sigma, leak=leak, comp=comp, w_input=w_input
        )

    # @torch.no_grad()
    def _recall(
            self, query_key,
            sigma=None, leak=None, comp=None, w_input=None
    ):
        """get the episodic memory according to some policy
        e.g. if the policy is 1nn, return the best matching memory
        e.g. the policy can be based on the rational model

        Parameters
        ----------
        query_key :

        Returns
        -------
        a row vector
            a DND value, representing the memory content
        """
        # compute similarity(query, memory_i ), for all i
        similarities_ = compute_similarities(
            query_key, self.keys, self.kernel,
            sigma=sigma
        )
        memory_wts = transform_similarities(
            similarities_, self.recall_func,
            leak=leak, comp=comp, w_input=w_input,
        ).view(1, -1)
        memory_matrix = list2mat(self.vals)
        retrieved_item = torch.mm(memory_wts, memory_matrix)
        return retrieved_item


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
        query_key, key_list, metric,
        sigma=None,
):
    """Compute the similarity between query vs. key_i for all i
        i.e. compute q M, w/ q: 1 x key_dim, M: #keys x key_dim

    Parameters
    ----------
    query_key : a row vector
        Description of parameter `query_key`.
    key_list : list
        Description of parameter `key_list`.
    metric : str
        Description of parameter `metric`.

    Returns
    -------
    a row vector w/ len #memories
        the similarity between query vs. key_i, for all i
    """
    if metric is 'rbf':
        assert sigma is not None, f'missing sigma for RBF computation'
    # reshape query to 1 x key_dim
    q = query_key.data.view(1, -1)
    # reshape memory keys to #keys x key_dim
    M = list2mat(key_list)
    # compute similarities
    if metric == 'cosine':
        similarities = F.cosine_similarity(q, M)
    # elif metric == 'corr':
    #     similarities = compute_correlation(q, M)
    elif metric == 'l1':
        similarities = - F.pairwise_distance(q, M, p=1)
    elif metric == 'l2':
        similarities = - F.pairwise_distance(q, M, p=2)
    elif metric == 'rbf':
        similarities = rbf(q, M, sigma)
    else:
        raise Exception(f'Unrecognizable metric: {metric}')
    return similarities


@torch.no_grad()
def empty_memory(memory_dim):
    """Get a empty memory, assuming the memory is a row vector
    """
    return torch.zeros(1, memory_dim).data
