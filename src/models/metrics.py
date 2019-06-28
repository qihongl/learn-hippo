import torch
import torch.nn.functional as F
from models.LCA_pytorch import LCA


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


def rbf(q, M, sigma):
    """compute RBF distance between q vs. all vectors in M

    Parameters
    ----------
    q : type
        Description of parameter `q`.
    M : type
        Description of parameter `M`.
    sigma : type
        Description of parameter `sigma`.

    Returns
    -------
    type
        Description of returned object.

    """
    similarities = torch.exp(
        -.5 * F.pairwise_distance(q, M, p=2)**2 / sigma ** 2
    )
    return similarities


def compute_correlation(q, M):
    """compute the correlation for a query key vs. all rows/memories in M

    Parameters
    ----------
    q : torch.FloatTensor, (1 x key_dim)
        the query key
    M : torch.FloatTensor, (key_dim x n_memories)
        the memory matrix

    Returns
    -------
    torch.FloatTensor, (n_memories,)
        the correlation for a query key vs. all rows/memories in M

    """
    corr_mat_ = corrcoef(torch.cat([q, M]))
    corr_qM = corr_mat_[1:, 0]
    return corr_qM


def corrcoef(x):
    """Mimics `np.corrcoef`
    Ref: https://github.com/pytorch/pytorch/issues/1254

    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c
