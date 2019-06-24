import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from itertools import product
from models.DND import compute_similarities, transform_similarities


def compute_trsm(activation_tensor):
    """compute TR-TR neural similarity for the input tensor

    Parameters
    ----------
    activation_tensor : 3d array, (n_examples, n_timepoints, n_dim)
        neural activity

    Returns
    -------
    2d array, (n_timepoints, n_timepoints)
        similarity array

    """
    n_examples, n_timepoints, n_dim = np.shape(activation_tensor)
    trsm_ = np.zeros((n_timepoints, n_timepoints))
    for data_i_ in activation_tensor:
        trsm_ += np.corrcoef(data_i_)
    return trsm_ / n_examples


def _compute_evidence(
    cell_states, memories, leak_, comp_, inpw_,
    mrwt_func, kernel,
):
    event_len, n_hidden = cell_states.size()
    evidence = np.zeros((event_len, len(memories)))
    for t in range(event_len):
        similarities_ = compute_similarities(
            cell_states[t, :], memories, kernel
        )
        evidence[t, :] = transform_similarities(
            similarities_, mrwt_func,
            leak=leak_[t], comp=comp_[t],
            w_input=inpw_[t],
        ).numpy()
    return evidence


def compute_evidence(
    C_tp, K_tp, Inpw_tp, Leak_tp, Comp_tp,
    mrwt_func, kernel
):
    n_mems = len(K_tp[0])
    n_trials_ = len(C_tp)
    event_len, n_hidden = C_tp[0].size()
    evidences_abs = np.zeros((event_len, n_mems, n_trials_))
    for i in range(n_trials_):
        # calculate the kernel-based similatity for target vs. lure
        evidences_abs[:, :, i] = _compute_evidence(
            C_tp[i], K_tp[i], Leak_tp[i], Comp_tp[i], Inpw_tp[i],
            mrwt_func, kernel
        )
    return evidences_abs


def compute_roc(distrib_noise, distrib_signal):
    """compute ROC given the two distribributions
    assuming the distributions are the output of np.histogram

    example:
    dist_l, _ = np.histogram(acts_l, bins=n_bins, range=histrange)
    dist_r, _ = np.histogram(acts_r, bins=n_bins, range=histrange)
    tprs, fprs = compute_roc(dist_l, dist_r)

    Parameters
    ----------
    distrib_noise : 1d array
        the noise distribution
    distrib_signal : 1d array
        the noise+signal distribution

    Returns
    -------
    1d array, 1d array
        the roc curve: true positive rate, and false positive rate

    """
    assert len(distrib_noise) == len(distrib_signal)
    assert np.sum(distrib_noise) == np.sum(distrib_signal)
    n_pts = len(distrib_noise)
    tpr, fpr = np.zeros(n_pts), np.zeros(n_pts)
    # slide the decision boundary from left to right
    for b in range(n_pts):
        fn, tp = np.sum(distrib_signal[:b]), np.sum(distrib_signal[b:])
        tn, fp = np.sum(distrib_noise[:b]), np.sum(distrib_noise[b:])
        # calculate TP rate and FP rate
        tpr[b] = tp / (tp + fn)
        fpr[b] = fp / (tn + fp)
    return tpr, fpr


def compute_auc_over_time(
        acts_l, acts_r,
        n_bins=100, histrange=(0, 1)
):
    """compute roc, auc, over time
    - given the activity for the two conditions
    - compute roc, auc for all time points
    *depends on analysis.neural.compute_roc()

    Parameters
    ----------
    acts_l : 2d array, (T x ?)
        the left distribution
    acts_r : 2d array, (T x ?)
        the right distribution
    n_bins : int
        histogram bin
    histrange : 2-tuple
        histogram range

    Returns
    -------
    arrays
        roc, auc, over time

    """
    event_len, n_examples = np.shape(acts_l)
    # compute fpr, tpr
    tprs = np.zeros((event_len, n_bins))
    fprs = np.zeros((event_len, n_bins))
    for t in range(event_len):
        # compute the bin counts for each condition
        dist_l, _ = np.histogram(acts_l[t, :], bins=n_bins, range=histrange)
        dist_r, _ = np.histogram(acts_r[t, :], bins=n_bins, range=histrange)
        tprs[t], fprs[t] = compute_roc(dist_l, dist_r)
    # compute area under roc cureves
    auc = [metrics.auc(fprs[t], tprs[t]) for t in range(event_len)]
    return tprs, fprs, auc
