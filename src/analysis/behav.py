import numpy as np
import torch
from scipy.stats import sem
from utils.utils import to_sqnp, to_np


def compute_acc(Y, log_dist_a, n_se=2, return_er=False):
    """compute the accuracy of the prediction, over time
    - optionally return the standard error
    - assume n_action == y_dim + 1

    Parameters
    ----------
    Y : 3d tensor
        [n_examples, T_total, y_dim]
    log_dist_a : 3d tensor
        [n_examples, T_total, n_action]
    n_se : int
        number of SE
    return_er : bool
        whether to return SEs

    Returns
    -------
    1d array(s)
        stats for state prediction accuracy

    """
    # argmax the action distribution (don't know unit included)
    argmax_dist_a = np.argmax(log_dist_a, axis=2)
    # argmax the targets one hot vecs
    argmax_Y = np.argmax(to_np(Y), axis=2)
    # compute matches
    corrects = argmax_Y == argmax_dist_a
    # compute stats across trials
    acc_mu_ = np.mean(corrects, axis=0)
    acc_er_ = sem(corrects, axis=0) * n_se
    if return_er:
        return acc_mu_, acc_er_
    return acc_mu_


def compute_dk(log_dist_a, n_se=2, return_er=False):
    """compute P(don't know) over time
    - optionally return the standard error
    - assume don't know is the last action dimension
    - assume n_action == y_dim + 1

    Parameters
    ----------
    log_dist_a : 3d tensor
        [n_examples, T_total, n_action]
    n_se : int
        number of SE
    return_er : bool
        whether to return SEs

    Returns
    -------
    1d array(s)
        stats for P(don't know)

    """
    a_dim = np.shape(log_dist_a)[-1]
    argmax_dist_a = np.argmax(log_dist_a, axis=2)
    dk = argmax_dist_a == (a_dim-1)
    dk_mu_ = np.mean(dk, axis=0)
    dk_er_ = sem(dk, axis=0)*n_se
    if return_er:
        return dk_mu_, dk_er_
    return dk_mu_


def average_by_part(time_course, p):
    """take average within each part of the (multi-part) sequence

    Parameters
    ----------
    time_course : 1d array
        a sequence of values; e.g. accuracy
    p : the param class
        simulation parameters

    Returns
    -------
    list
        a list of averaged values

    """
    return [np.mean(time_course[get_tps_for_ith_part(ip, p.env.tz.T_part)])
            for ip in range(p.env.tz.n_mvs)]


def get_tps_for_ith_part(ip, T_part):
    """get the time range (a list of time points) for the i-th movie part

    Parameters
    ----------
    ip : int
        the index of movie part
    T_part : int
        the length of one movie part

    Returns
    -------
    1d array
        a range of time points

    """
    return np.arange(T_part*ip, T_part*(ip+1))


def entropy(probs):
    """calculate entropy.
    I'm using log base 2!

    Parameters
    ----------
    probs : a torch vector
        a prob distribution

    Returns
    -------
    torch scalar
        the entropy of the distribution

    """
    return - torch.stack([pi * torch.log2(pi) for pi in probs]).sum()


def get_baseline(T, chance):
    """compute the observation-only (no memory) baseline performance

    Parameters
    ----------
    T : int
        event length
    chance : float [0,1]
        chance performance, 1 / n branches

    Returns
    -------
    np.array (T+1,)
        baseline performance accuracy

    """
    return np.array([chance * (T-t)/T + t/T for t in range(T+1)])


# def compute_performance_metrics(Y, action_distribution, p):
#     """compute performance metrics.
#
#     e.g.
#     [X, Y], _ = get_data_tz(n_examples, p)
#     log_loss_i, log_return_i, log_adist_i = run_exp_tz(
#         agent, optimizer, X, Y, p, supervised,
#         learning=True
#     )
#     pm_ = compute_performance_metrics(Y, log_adist_i, p)
#     corrects_mu_, dk_probs_mu_, mistakes_mu_ = pm_
#
#     Parameters
#     ----------
#     Y : type
#         Description of parameter `Y`.
#     action_distribution : type
#         Description of parameter `action_distribution`.
#     p : type
#         Description of parameter `p`.
#
#     Returns
#     -------
#     1d array, 1d array, float
#         performance metrics
#
#     """
#     # compute correct rate
#     corrects = compute_predacc(to_sqnp(Y), action_distribution)
#     dks = compute_dks(action_distribution)
#     # compute mus
#     corrects_mu_ = np.mean(corrects, axis=0)
#     dk_probs_mu_ = np.mean(dks, axis=0)
#     # remove event end points
#     corrects_mu__ = np.delete(corrects_mu_, p.env.tz.event_ends)
#     dk_probs_mu__ = np.delete(dk_probs_mu_, p.env.tz.event_ends)
#     mistakes_ = np.ones_like(corrects_mu__) - corrects_mu__ - dk_probs_mu__
#     mistakes_mu_ = np.sum(mistakes_)
#     return corrects_mu_, dk_probs_mu_, mistakes_mu_
