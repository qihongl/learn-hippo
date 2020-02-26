import os
import numpy as np
from utils.io import pickle_load_dict
from utils.constants import TZ_COND_DICT
from analysis import compute_stats, remove_none
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='poster')

# constants
lca_pnames = {0: 'input gate', 1: 'competition'}
all_conds = list(TZ_COND_DICT.values())
T = 16

gdata_outdir = 'temp/'
penaltys_train = [4]
penaltys_test = [0, 4]

'''load data'''
lca_param = {ptest: None for ptest in penaltys_test}
auc = {ptest: None for ptest in penaltys_test}
acc = {ptest: None for ptest in penaltys_test}
mis = {ptest: None for ptest in penaltys_test}
dk = {ptest: None for ptest in penaltys_test}

# for ptrain in penaltys_train:
ptrain = penaltys_train[0]
for ptest in penaltys_test:
    print(f'ptrain={ptrain}, ptest={ptest}')
    # load data
    fname = f'p{ptrain}-{ptest}-data.pkl'
    data_load_path = os.path.join(gdata_outdir, fname)
    data = pickle_load_dict(data_load_path)
    # unpack data
    lca_param[ptest] = data['lca_param_dicts']
    auc[ptest] = data['auc_list']
    acc[ptest] = data['acc_dict']
    mis[ptest] = data['mis_dict']
    dk[ptest] = data['dk_dict']

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
            for lca_pid, lca_pname in lca_pnames.items():
                del lca_param[ptest][lca_pid][cond]['mu'][i_ms]
                del lca_param[ptest][lca_pid][cond]['er'][i_ms]


'''process the data: extract differences between the two penalty conds'''

ptest1 = 0
ptest2 = 4


def extract_part2_diff(val, cond):
    tmp = np.array(val[ptest2][cond]['mu']) - \
        np.array(val[ptest1][cond]['mu'])
    return tmp[:, T:]


# extract differences
rt = {ptest: None for ptest in penaltys_test}
for ptest in penaltys_test:
    rt_ = np.array(lca_param[ptest][0]['DM']['mu'])[:, T:].T
    rt_ / np.sum(rt_)
    rt[ptest] = np.mean(rt_ * np.reshape(np.arange(T) + 1, (T, 1)), axis=0)


lca_param_diff = {
    lca_pname_: {
        cond: np.zeros((n_subjs, T)) for cond in all_conds
    }
    for lca_pname_ in lca_pnames.values()
}
# auc_diff = {cond: np.zeros((n_subjs, T)) for cond in all_conds}
acc_diff = {cond: np.zeros((n_subjs, T)) for cond in all_conds}
mis_diff = {cond: np.zeros((n_subjs, T)) for cond in all_conds}
dk_diff = {cond: np.zeros((n_subjs, T)) for cond in all_conds}

auc_diff = np.array(auc[ptest2]) - np.array(auc[ptest1])
for cond in all_conds:
    acc_diff[cond] = extract_part2_diff(acc, cond)
    mis_diff[cond] = extract_part2_diff(mis, cond)
    dk_diff[cond] = extract_part2_diff(dk, cond)
    for lca_pid, lca_pname in lca_pnames.items():
        tmp = np.array(lca_param[ptest2][lca_pid][cond]['mu']) - \
            np.array(lca_param[ptest1][lca_pid][cond]['mu'])
        lca_param_diff[lca_pname][cond] = tmp[:, T:]


'''regression models'''

# auc ~ change in input gate and competition
cond = 'DM'
f, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for lca_pid, lca_pname in lca_pnames.items():
    iv_ = np.mean(lca_param_diff[lca_pname][cond], axis=1)
    sns.regplot(iv_, auc_diff, 'x', ax=axes[lca_pid])
    axes[lca_pid].set_ylabel(r'$\Delta$ AUC')
    axes[lca_pid].set_xlabel(r'$\Delta$ %s' % (lca_pname))
sns.despine()
f.tight_layout()

# auc ~ change in input gate x competition
iv_ = np.ones(shape=np.shape(iv_))
for lca_pid, lca_pname in lca_pnames.items():
    iv_ *= np.mean(lca_param_diff[lca_pname][cond], axis=1)

r_val, p_val = pearsonr(iv_, auc_diff)
f, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.regplot(iv_, auc_diff, 'x')
ax.set_ylabel(r'$\Delta$ AUC')
ax.set_xlabel(r'$\Delta$ interaction')
ax.annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val), xy=(
    0.05, 0.05), xycoords='axes fraction')
sns.despine()
f.tight_layout()

# auc ~ recall time (center of mass of input gate)
rt_diff = rt[ptest2] - rt[ptest1]
r_val, p_val = pearsonr(rt_diff, auc_diff)
f, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.regplot(rt_diff, auc_diff, 'x')
ax.set_ylabel(r'$\Delta$ AUC')
ax.set_xlabel(r'$\Delta$ RT')
ax.annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val), xy=(
    0.05, 0.05), xycoords='axes fraction')
sns.despine()
f.tight_layout()


# mistakes ~ change in input gate and competition
cond = 'DM'
diff_data = {
    'Accuracy': acc_diff, 'Mistakes': mis_diff, 'Don\'t know': dk_diff
}

for data_name, diff_data_ in diff_data.items():
    dv_ = np.mean(diff_data_[cond], axis=1)
    f, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for lca_pid, lca_pname in lca_pnames.items():
        iv_ = np.mean(lca_param_diff[lca_pname][cond], axis=1)
        r_val, p_val = pearsonr(iv_, dv_)
        sns.regplot(iv_, dv_, 'x', ax=axes[lca_pid])
        axes[lca_pid].set_ylabel(r'$\Delta$ %s' % (data_name))
        axes[lca_pid].set_xlabel(r'$\Delta$ %s' % (lca_pname))
        axes[lca_pid].annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val), xy=(
            0.05, 0.05), xycoords='axes fraction')
    sns.despine()
    f.tight_layout()


dv_ = np.mean(acc_diff[cond], axis=1)
r_val, p_val = pearsonr(iv_, dv_)

f, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for lca_pid, lca_pname in lca_pnames.items():
    iv_ = np.mean(lca_param_diff[lca_pname][cond], axis=1)
    sns.regplot(iv_, dv_, 'x', ax=axes[lca_pid])
    axes[lca_pid].set_ylabel(r'$\Delta$ AUC')
    axes[lca_pid].set_xlabel(r'$\Delta$ %s' % (lca_pname))
    axes[lca_pid].annotate(r'$r \approx %.2f$, $p \approx %.2f$' % (r_val, p_val), xy=(
        0.05, 0.05), xycoords='axes fraction')
sns.despine()
f.tight_layout()
