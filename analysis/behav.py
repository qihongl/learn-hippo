import numpy as np
from scipy.stats import sem
from utils.utils import to_sqnp, to_np


def compute_acc(Y, log_dist_a, n_se=2, return_er=False):
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
    a_dim = np.shape(log_dist_a)[-1]
    argmax_dist_a = np.argmax(log_dist_a, axis=2)
    dk = argmax_dist_a == a_dim
    dk_mu_ = np.mean(dk, axis=0)
    dk_er_ = sem(dk, axis=0)*n_se
    if return_er:
        return dk_mu_, dk_er_
    return dk_mu_


def average_by_part(time_course, p):
    return [np.mean(time_course[get_tps_for_ith_part(ip, p.env.tz.T_part)])
            for ip in range(p.env.tz.n_mvs)]


def get_tps_for_ith_part(ip, T_part):
    return np.arange(T_part*ip, T_part*(ip+1))

# def compute_dks(action_distribution):
#     """compute the don't know indicator matrix
#
#     Assumptions:
#     - don't know dimension is the last dimension in action space
#
#     Parameters
#     ----------
#     action_distribution : 3d array (n_examples, T, vec_dim)
#         probability distribution over actions
#
#     Returns
#     -------
#     dks, 2d array (n_examples, T)
#         binary matrix where 1 at i,t position means the model said don't know
#         at time t in trial i
#
#     """
#     n_examples, T, vec_dim = np.shape(action_distribution)
#     dks = np.array([
#         np.argmax(action_distribution[i, :, :], axis=1) == vec_dim-1
#         for i in range(n_examples)
#     ])
#     return dks
#
#
# def compute_predacc(Y, Yhat):
#     """compute the prediction performance
#
#     Parameters
#     ----------
#     Y : np.array, (n_examples, total_event_len, ohv_dim)
#         target, states vecs
#     Yhat : np.array, (n_examples, total_event_len, ohv_dim)
#         predicted states vecs
#
#     Returns
#     -------
#     np.array (n_examples, total_event_len)
#         corrects
#
#     """
#     n_examples, total_event_len, ohv_dim = np.shape(Y)
#     corrects = np.zeros((n_examples, total_event_len))
#     for m in range(n_examples):
#         state_predictions = np.argmax(Yhat[m], axis=1)
#         actual_states = np.argmax(Y[m], axis=1)
#         corrects[m, :] = np.array(
#             state_predictions == actual_states, dtype=float
#         )
#     return corrects
#
#
# def get_baseline(T, chance):
#     """compute the observation-only (no memory) baseline performance
#
#     Parameters
#     ----------
#     T : int
#         event length
#     chance : float [0,1]
#         chance performance, 1 / n branches
#
#     Returns
#     -------
#     np.array (T+1,)
#         baseline performance accuracy
#
#     """
#     return np.array([chance * (T-t)/T + t/T for t in range(T+1)])
#
#
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
