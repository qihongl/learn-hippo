import numpy as np
# import pandas as pd
# import torch.nn as nn

# from sklearn import metrics
# from itertools import product
# from models.DND import compute_similarities, transform_similarities
# from analysis import compute_predacc
# from utils_analysis import compute_correct_rate
# from dep.qmvpa.signal import compute_roc

# softmax = nn.Softmax()

"""data org
"""


# def extrat_trials(data_list, trial_ids):
#     # trial_ids is 1 based
#     return [data_list[i] for i in trial_ids]
#
#
# def form_df(input_matrix):
#     n_examples, n_time_steps = np.shape(input_matrix)
#     # init df
#     col_labels = ['sim_id', 'time', 'val']
#     df = pd.DataFrame(columns=col_labels)
#     # # concate to get df
#     for i in range(n_examples):
#         sim_id = np.repeat(i, n_time_steps)
#         df = df.append(
#             pd.DataFrame(
#                 np.vstack([sim_id, range(n_time_steps), input_matrix[i, :]]).T,
#                 columns=col_labels
#             )
#         )
#     return df
#
#
# def form_evd_df_2mem(evidences):
#     assert np.shape(evidences)[1] == 2
#     # convert evidence matrices to dfs
#     evidences0_df = form_df(evidences[:, 0, :].T)
#     evidences1_df = form_df(evidences[:, 1, :].T)
#     # attach condition labels
#     evidences0_df['condition'] = pd.Series(['target'] * len(evidences0_df))
#     evidences1_df['condition'] = pd.Series(['lure'] * len(evidences1_df))
#     # form the join df
#     evidences_df = pd.concat([evidences0_df, evidences1_df])
#     return evidences_df
#
#
# """performance computation
# """
#
#
# def _compute_evidence(
#     cell_states, memories, leak_, comp_, inpw_,
#     mrwt_func, kernel,
# ):
#     event_len, n_hidden = cell_states.size()
#     evidence = np.zeros((event_len, len(memories)))
#     for t in range(event_len):
#         similarities_ = compute_similarities(
#             cell_states[t, :], memories, kernel
#         )
#         evidence[t, :] = transform_similarities(
#             similarities_, mrwt_func,
#             leak=leak_[t], comp=comp_[t],
#             w_input=inpw_[t],
#         ).numpy()
#     return evidence
#
#
# def compute_evidence(
#     C_tp, K_tp, Inpw_tp, Leak_tp, Comp_tp,
#     mrwt_func, kernel
# ):
#     n_mems = len(K_tp[0])
#     n_trials_ = len(C_tp)
#     event_len, n_hidden = C_tp[0].size()
#     evidences_abs = np.zeros((event_len, n_mems, n_trials_))
#     for i in range(n_trials_):
#         # calculate the kernel-based similatity for target vs. lure
#         evidences_abs[:, :, i] = _compute_evidence(
#             C_tp[i], K_tp[i], Leak_tp[i], Comp_tp[i], Inpw_tp[i],
#             mrwt_func, kernel
#         )
#     return evidences_abs
#
#
# def process_trial_type_info(trial_type_info_):
#     # get all trial types
#     trial_types = np.unique(trial_type_info_)
#     n_trial_types = len(trial_types)
#     # get trial selector
#     trial_ids_set = [np.where(np.array(trial_type_info_) == k)[0]
#                      for k in trial_types]
#     # count n trial per type
#     n_trials = [len(trial_ids_) for trial_ids_ in trial_ids_set]
#     return trial_types, n_trial_types, trial_ids_set, n_trials
#
#
# def compute_baseline(T, chance):
#     """compute the observation-only (no memory) baseline performance
#     """
#     return np.array([chance * (T-t)/T + t/T for t in range(T)])
#
#
# def compute_memory_benefit(
#         correct_rates_mat,
#         n_movies_in_seq, event_len, n_branches,
#         return_normalized=True
# ):
#     """compute the the gain w.r.t to the observation only baseline
#     """
#     shift_ = event_len * n_movies_in_seq
#     # average accs
#     mean_acc_over_time = np.mean(correct_rates_mat, axis=0)[shift_:]
#     # compute baseline
#     chance = 1/n_branches
#     normalized_mem_benefit = compute_mb(mean_acc_over_time, chance)
#     # baseline = compute_baseline(len(mean_acc_over_time), chance)
#     # # compute net memory benefit
#     # mem_benefit = mean_acc_over_time - baseline
#     # # normalize by achievable mb
#     # total_mb = np.mean(np.ones_like(baseline) - baseline)
#     # normalized_mem_benefit = np.mean(mem_benefit) / total_mb
#     return normalized_mem_benefit
#
#
# def compute_mb(mean_acc_over_time, chance):
#     baseline = compute_baseline(len(mean_acc_over_time), chance)
#     # compute net memory benefit
#     mem_benefit = mean_acc_over_time - baseline
#     # normalize by achievable mb
#     total_mb = np.mean(np.ones_like(baseline) - baseline)
#     normalized_mem_benefit = np.mean(mem_benefit) / total_mb
#     return normalized_mem_benefit
#
#
# def compute_correct_rate_wrapper(Y_test, Y_hat, selected_trial_ids):
#     # compute the performance
#     Y_test_selected = Y_test[selected_trial_ids, :, :]
#     Y_test_selected = np.squeeze(Y_test_selected)
#     Y_hat_selected = Y_hat[selected_trial_ids, :, :]
#     # compute the rate
#     accs = compute_predacc(Y_test_selected, Y_hat_selected)
#     return accs
#
#
# # def compute_memory_benefit_wrapper(
# #         cache_pack, n_movies_in_seq, event_len, n_branches,
# #         trial_type_info
# # ):
# #     Y_test = cache_pack[0]
# #     Y_hat = cache_pack[1]
# #     # preproc results
# #     Y_test = Y_test.numpy()
# #     Y_hat = Y_hat.data.numpy()
# #     trial_type_info_ = process_trial_type_info(trial_type_info)
# #     [trial_types, n_trial_types, trial_ids_set, n_trials] = trial_type_info_
# #     # compute correct rates
# #     corrects = [
# #         compute_correct_rate_wrapper(Y_test, Y_hat, trial_ids_)
# #         for trial_ids_ in trial_ids_set
# #     ]
# #     # compute correct rates
# #     mbs = [
# #         compute_memory_benefit(
# #             corrects_, n_movies_in_seq, event_len, n_branches,
# #             return_normalized=True)
# #         for corrects_ in corrects
# #     ]
# #     return mbs
# #
# #
# # def compute_memory_benefit_wrapper_(
# #         cache_pack, event_len, n_branches, trial_type_info
# # ):
# #     Y_test = cache_pack[0]
# #     Y_hat = cache_pack[1]
# #     # preproc results
# #     Y_test = Y_test.numpy()
# #     Y_hat = Y_hat.data.numpy()
# #     trial_type_info_ = process_trial_type_info(trial_type_info)
# #     [trial_types, n_trial_types, trial_ids_set, n_trials] = trial_type_info_
# #     # compute correct rates
# #     corrects = [
# #         compute_correct_rate_wrapper(Y_test, Y_hat, trial_ids_)
# #         for trial_ids_ in trial_ids_set
# #     ]
# #     # compute correct rates
# #     n_movies_in_seq = 0
# #     mbs = [
# #         compute_memory_benefit(
# #             corrects_, n_movies_in_seq, event_len, n_branches,
# #             return_normalized=True)
# #         for corrects_ in corrects
# #     ]
# #     return mbs
#
#
# """RDM"""
#
#
# def compute_similarity_matrix(C_trial, sigma_trial):
#     T, _ = C_trial.size()
#     sim_mat = np.zeros((T, T))
#     for i, j in product(range(T), range(T)):
#         if i >= j:
#             sigma_ = sigma_trial[i]
#         else:
#             sigma_ = sigma_trial[j]
#         sim_mat[i, j] = compute_similarities(
#             C_trial[i], [C_trial[j]], 'rbf', sigma_,
#         ).item()
#     return sim_mat
#
#
# def compute_similarities_for_all_trials(Sigma_selected_, C_selected_):
#     n_trials_, T_ = np.shape(Sigma_selected_)
#     # compute the similarity matrix for all trials
#     sim_mats = np.zeros((T_, T_, n_trials_))
#     corr_mats = np.zeros((T_, T_, n_trials_))
#     for trial_id in range(len(C_selected_)):
#         C_trial = C_selected_[trial_id]
#         # compute LSTM kernel based similarity
#         sigma_trial = Sigma_selected_[trial_id]
#         sim_mats[:, :, trial_id] = compute_similarity_matrix(
#             C_trial, sigma_trial)
#         # compute LSTM indep corr similarity
#         corr_mats[:, :, trial_id] = np.corrcoef(C_trial.data.numpy())
# #         corr_mats[:,:,trial_id] = -squareform(pdist(C_trial.data.numpy()))
#
#     # compute the average
#     sim_mats_mean = np.mean(sim_mats, axis=2)
#     sim_mats_std = np.std(sim_mats, axis=2)
#     corr_mats_mean = np.mean(corr_mats, axis=2)
#     corr_mats_std = np.std(corr_mats, axis=2)
#     return sim_mats_mean, sim_mats_std, corr_mats_mean, corr_mats_std
#
#
# def compute_rt(corrects_, rt_bound, event_len):
#     for t in range(event_len - rt_bound):
#         if np.all(corrects_[t: t+rt_bound] == 1):
#             return t
#     return event_len - rt_bound
#
#
# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
#
# # def compute_auc_over_time(
# #         acts_l, acts_r,
# #         n_bins=100, histrange=(0, 1)
# # ):
# #     """compute roc, auc
# #     - given the activity for the two conditions
# #     - compute roc, auc for all time points
# #     """
# #     event_len, n_examples = np.shape(acts_l)
# #     # compute fpr, tpr
# #     tprs, fprs = np.zeros((event_len, n_bins)), np.zeros(
# #         (event_len, n_bins))
# #     for t in range(event_len):
# #         # compute the bin counts for each condition
# #         dist_l, _ = np.histogram(acts_l[t, :], bins=n_bins, range=histrange)
# #         dist_r, _ = np.histogram(acts_r[t, :], bins=n_bins, range=histrange)
# #         tprs[t], fprs[t] = compute_roc(dist_l, dist_r)
# #     # compute area under roc cureves
# #     auc = [metrics.auc(fprs[t], tprs[t]) for t in range(event_len)]
# #     return tprs, fprs, auc
