import dabest
import pandas as pd
import os
import numpy as np
from utils.params import P
from utils.io import pickle_load_dict, build_log_path
from utils.constants import TZ_COND_DICT
from analysis import compute_stats, remove_none
from scipy.stats import pearsonr
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='poster')
log_root = '../log/'
# constants
lca_pnames = {0: 'input gate', 1: 'competition'}
all_conds = list(TZ_COND_DICT.values())
T = 16

exp_name = 'vary-training-penalty'
gdata_outdir = 'data/'

penaltys_train = [0, 4]
penaltys_test = [0, 4]


n_param = 16
n_subjs = 15
subj_ids = np.arange(n_subjs)
penalty_random = 0
def_prob = .25
n_def_tps = 0
comp_val = .8
# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0
attach_cond = 0
supervised_epoch = 600
epoch_load = 1000
learning_rate = 7e-4
n_branch = 4
n_param = 16
enc_size = 16
n_event_remember = 2
similarity_max_test = .9
similarity_min_test = 0
n_examples_test = 256


'''load data'''
lca_param = {ptest: None for ptest in penaltys_test}
auc = {ptest: None for ptest in penaltys_test}
acc = {ptest: None for ptest in penaltys_test}
mis = {ptest: None for ptest in penaltys_test}
dk = {ptest: None for ptest in penaltys_test}
ma_lca = defaultdict()
ma_cosine = defaultdict()

# for ptrain in penaltys_train:
# ptrain = penaltys_train[0]
for ptrain, ptest in zip(penaltys_train, penaltys_test):
    # print(f'ptrain={ptrain}, ptest={ptest}')
    # # load data
    # fname = '%s-dp%.2f-p%d-%d.pkl' % (
    #     exp_name, def_prob, ptrain, ptest)
    # data_load_path = os.path.join(gdata_outdir, fname)
    # data = pickle_load_dict(data_load_path)

    p = P(
        exp_name=exp_name, sup_epoch=supervised_epoch,
        n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
        enc_size=enc_size, n_event_remember=n_event_remember,
        def_prob=def_prob, n_def_tps=n_def_tps,
        penalty=ptrain, penalty_random=penalty_random,
        p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
        cmpt=comp_val,
    )

    # create logging dirs
    log_path, _ = build_log_path(
        0, p, log_root=log_root, mkdir=False, verbose=False
    )
    # load data
    dir_all_subjs = os.path.dirname(log_path)
    fname = 'p%d-%d.pkl' % (ptrain, ptest)
    data = pickle_load_dict(os.path.join(dir_all_subjs, fname))

    # unpack data
    lca_param[ptest] = data['lca_param_dicts']
    auc[ptest] = data['auc_list']
    acc[ptest] = data['acc_dict']
    mis[ptest] = data['mis_dict']
    dk[ptest] = data['dk_dict']
    ma_lca[ptest] = data['lca_ma_list']
    ma_cosine[ptest] = data['cosine_ma_list']

n_subjs_total = len(auc[ptest])

# process the data - identify missing subjects
missing_subjects = []
for ptest in penaltys_test:
    for lca_pid, lca_pname in lca_pnames.items():
        for cond in all_conds:
            _, missing_ids_ = remove_none(
                lca_param[ptest][lca_pid][cond]['mu'],
                return_missing_idx=True
            )
            missing_subjects.extend(missing_ids_)
missing_subjects = np.unique(missing_subjects)

n_subjs = n_subjs_total - len(missing_subjects)

# process the data - remove missing subjects for all data dicts
for i_ms in sorted(missing_subjects, reverse=True):
    # print(i_ms)
    for ptest in penaltys_test:
        del auc[ptest][i_ms]
        for cond in all_conds:
            del acc[ptest][cond]['mu'][i_ms]
            del mis[ptest][cond]['mu'][i_ms]
            del dk[ptest][cond]['mu'][i_ms]
            del acc[ptest][cond]['er'][i_ms]
            del mis[ptest][cond]['er'][i_ms]
            del dk[ptest][cond]['er'][i_ms]
            # del ma_lca_dm[ptest][i_ms]
            for lca_pid, lca_pname in lca_pnames.items():
                del lca_param[ptest][lca_pid][cond]['mu'][i_ms]
                del lca_param[ptest][lca_pid][cond]['er'][i_ms]


'''process the data: extract differences between the two penalty conds'''


def extract_part2_diff(val, cond):
    tmp = np.array(val[ptest2][cond]['mu']) - \
        np.array(val[ptest1][cond]['mu'])
    return tmp[:, T:]


ptest1 = penaltys_test[0]
ptest2 = penaltys_test[1]

# extract differences
rt = {ptest: None for ptest in penaltys_test}
time_vector = np.reshape(np.arange(T) + 1, (T, 1))
for ptest in penaltys_test:
    ig_p2_ = np.array(lca_param[ptest][0]['DM']['mu'])[:, T:].T
    ig_p2_norm = ig_p2_ / np.sum(ig_p2_, axis=0)
    # ig_p2_norm = ig_p2_
    # np.mean(ig_p2_norm, axis=1)
    # rt[ptest] = np.mean(ig_p2_ * time_vector, axis=0)
    rt[ptest] = np.reshape(np.dot(ig_p2_norm.T, time_vector), (-1,))
    # rt[ptest] = np.mean(rt_all_subjs)


# lca_param_diff = {
#     lca_pname_: {
#         cond: np.zeros((n_subjs, T)) for cond in all_conds
#     }
#     for lca_pname_ in lca_pnames.values()
# }
# # auc_diff = {cond: np.zeros((n_subjs, T)) for cond in all_conds}
# acc_diff = {cond: np.zeros((n_subjs, T)) for cond in all_conds}
# mis_diff = {cond: np.zeros((n_subjs, T)) for cond in all_conds}
# dk_diff = {cond: np.zeros((n_subjs, T)) for cond in all_conds}
#
# auc_diff = np.array(auc[ptest2]) - np.array(auc[ptest1])
# for cond in all_conds:
#     acc_diff[cond] = extract_part2_diff(acc, cond)
#     mis_diff[cond] = extract_part2_diff(mis, cond)
#     dk_diff[cond] = extract_part2_diff(dk, cond)
#     for lca_pid, lca_pname in lca_pnames.items():
#         tmp = np.array(lca_param[ptest2][lca_pid][cond]['mu']) - \
#             np.array(lca_param[ptest1][lca_pid][cond]['mu'])
#         lca_param_diff[lca_pname][cond] = tmp[:, T:]
#
# rt_diff = rt[ptest2] - rt[ptest1]


# def compute_reward(ptest_):
#     cond = 'DM'
#     acc_mu_p2 = np.array(acc[ptest_][cond]['mu'])[:, T:]
#     mis_mu_p2 = np.array(mis[ptest_][cond]['mu'])[:, T:]
#     reward_ptest_ = np.sum(acc_mu_p2, axis=1) - \
#         np.sum(mis_mu_p2, axis=1) * ptest_
#     return reward_ptest_
#
#
# reward = {ptest: compute_reward(ptest) for ptest in penaltys_test}
# reward_diff = reward[ptest2] - reward[ptest1]

data_dict = {'Penalty low': rt[0], 'Penalty high': rt[4]}
df = pd.DataFrame(data_dict)
df['ids'] = np.arange(n_subjs)
df.head()

# Load the data into dabest
dabest_data = dabest.load(
    data=df, idx=list(data_dict.keys()), paired=True, id_col='ids'
)
dabest_data.mean_diff.plot(swarm_label='Recall time', fig_size=(11, 6))
print(dabest_data.mean_diff)
dabest_data.mean_diff.statistical_tests
