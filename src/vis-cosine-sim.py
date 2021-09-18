import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.io import pickle_load_dict
from analysis import compute_stats, remove_none
from utils.constants import TZ_COND_DICT
sns.set(style='white', palette='colorblind', context='poster')
all_cond = TZ_COND_DICT.values()


gdata_outdir = 'data/'
exp_name = 'vary-test-penalty'
def_prob = .25
penalty_train = 4
penalty_test_list = [0, 2, 4]
T = 16

# f, ax = plt.subplots(1, 1, figsize=(6, 5))
ma_cos_mumu = [np.zeros(len(all_cond)) for _ in range(len(penalty_test_list))]
ma_cos_muse = [np.zeros(len(all_cond)) for _ in range(len(penalty_test_list))]
ma_cos_tmu = [np.zeros((len(all_cond), T))
              for _ in range(len(penalty_test_list))]
ma_cos_tse = [np.zeros((len(all_cond), T))
              for _ in range(len(penalty_test_list))]
memory_sim_mu = np.zeros(len(penalty_test_list))
memory_sim_se = np.zeros(len(penalty_test_list))

for p_i, penalty_test in enumerate(penalty_test_list):
    fname = '%s-dp%.2f-p%d-%d.pkl' % (
        exp_name, def_prob, penalty_train, penalty_test)

    data = pickle_load_dict(os.path.join(gdata_outdir, fname))
    ma_cos_list = data['cosine_ma_list']
    memory_sim_g = data['memory_sim_g']

    '''group level memory activation by condition, averaged over time'''

    ma_cos_list_nonone = remove_none(ma_cos_list)
    n_actual_subjs = len(ma_cos_list_nonone)
    ma_cos_mu = {
        cond: np.zeros(n_actual_subjs,) for cond in all_cond
    }
    ma_cos = {
        cond: np.zeros((n_actual_subjs, T)) for cond in all_cond
    }
    for i_s in range(n_actual_subjs):
        for c_i, c_name in enumerate(all_cond):
            # average over time
            ma_cos[c_name][i_s] = ma_cos_list_nonone[i_s][c_name]['lure']['mu'][T:]
            ma_cos_mu[c_name][i_s] = np.mean(
                ma_cos_list_nonone[i_s][c_name]['lure']['mu'][T:]
            )

    # compute stats across subjects
    for c_i, c_name in enumerate(all_cond):
        ma_cos_mumu[p_i][c_i], ma_cos_muse[p_i][c_i] = compute_stats(
            ma_cos_mu[c_name])
        ma_cos_tmu[p_i][c_i], ma_cos_tse[p_i][c_i] = compute_stats(
            ma_cos[c_name])

    # compute memory similarity
    memory_sim_mu[p_i], memory_sim_se[p_i] = compute_stats(
        remove_none(memory_sim_g)
    )

'''plot the data'''

cpal = sns.color_palette("Blues", n_colors=len(all_cond))
f, ax = plt.subplots(1, 1, figsize=(6, 5))
for p_i, penalty_test in enumerate(penalty_test_list):
    ax.errorbar(
        x=range(len(all_cond)), y=ma_cos_mumu[p_i], yerr=ma_cos_muse[p_i],
        color=cpal[p_i]
    )
ax.set_ylabel('Cosine similarity')
ax.set_xlabel('Condition')
ax.set_xticks(range(len(all_cond)))
ax.set_xticklabels(all_cond)
ax.set_ylim([.1, .8])
ax.legend(penalty_test_list, title='penalty test')
sns.despine()

f, axes = plt.subplots(1, len(all_cond), figsize=(15, 5), sharey=True)
for c_i, c_name in enumerate(all_cond):
    for p_i, penalty_test in enumerate(penalty_test_list):
        axes[c_i].errorbar(
            x=range(T), y=ma_cos_tmu[p_i][c_i], yerr=ma_cos_tse[p_i][c_i],
            color=cpal[p_i]
        )
    axes[c_i].set_title(c_name)
    axes[c_i].set_ylabel('Cosine similarity')
    axes[c_i].set_xlabel('Time (part 2)')
    axes[c_i].set_ylim([-.1, .9])
    axes[c_i].axhline(0, linestyle='--', color='grey')
axes[0].legend(penalty_test_list, title='penalty test')
sns.despine()
f.tight_layout()

f, ax = plt.subplots(1, 1, figsize=(6, 5))
for p_i, penalty_test in enumerate(penalty_test_list):
    ax.errorbar(
        x=range(len(penalty_test_list)), y=memory_sim_mu, yerr=memory_sim_mu,
    )
ax.set_xticks(range(len(penalty_test_list)))
ax.set_xticklabels(penalty_test_list)
ax.set_ylabel('Cosine similarity')
ax.set_xlabel('Penalty')
ax.set_ylim([-.1, .8])
sns.despine()
