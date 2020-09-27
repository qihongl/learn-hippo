import os
import numpy as np
from utils.io import pickle_load_dict
from analysis import compute_stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='poster')


def remove_all_zero_rows(data_mat):
    data_mat = data_mat[~np.all(data_mat == 0, axis=1)]
    data_mat = data_mat[~np.any(np.isnan(data_mat), axis=1)]
    return data_mat


data_name = 'CM_p1'
# data_name = 'DA_p1'
# data_name = 'DA_p2'
def_prob_range = np.arange(.25, 1, .1)
# def_prob_range = np.array(list(def_prob_range) + [.95])


norm_mu = np.zeros((len(def_prob_range), 3))
norm_se = np.zeros((len(def_prob_range), 3))
d_norm_mu = np.zeros((len(def_prob_range), 3))
d_norm_se = np.zeros((len(def_prob_range), 3))
delta_norm_mu = np.zeros((len(def_prob_range), 2))
delta_norm_se = np.zeros((len(def_prob_range), 2))

for dpi, def_prob in enumerate(def_prob_range):

    data_dict = pickle_load_dict(
        f'temp/dist-%s-%.2f.pkl' % (data_name, def_prob))
    # if dpi == len(def_prob_range):
    #     data_dict = pickle_load_dict(f'temp/dist-%.2f.pkl' % (def_prob))

    norm_data_g = remove_all_zero_rows(data_dict['norm_data_g'])
    d_norm_g = remove_all_zero_rows(data_dict['d_norm_g'])
    delta_norm_g = remove_all_zero_rows(data_dict['delta_norm_g'])

    norm_mu[dpi], norm_se[dpi] = compute_stats(norm_data_g)
    d_norm_mu[dpi], d_norm_se[dpi] = compute_stats(d_norm_g)
    delta_norm_mu[dpi], delta_norm_se[dpi] = compute_stats(delta_norm_g)


legends = ['S', 'NS', 'dk']
f, ax = plt.subplots(1, 1, figsize=(8, 5))
xticklabels = ['%.2f' % dp for dp in def_prob_range]
xticks = range(len(def_prob_range))
for i, leg in enumerate(legends):
    ax.errorbar(xticks, norm_mu[:, i], yerr=norm_se[:, i], label=leg)
ax.set_title(data_name)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_ylabel('Activity norm')
ax.legend()
f.tight_layout()
sns.despine()

fig_path = os.path.join(f'temp/norm-%s.png' % (data_name))
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

legends = ['dk - S', 'dk - NS', 'S - NS']
f, ax = plt.subplots(1, 1, figsize=(8, 5))
xticklabels = ['%.2f' % dp for dp in def_prob_range]
xticks = range(len(def_prob_range))
for i, leg in enumerate(legends):
    ax.errorbar(xticks, d_norm_mu[:, i], yerr=d_norm_se[:, i], label=leg)
ax.set_title(data_name)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_ylabel('Distance')
ax.legend()
f.tight_layout()
sns.despine()
fig_path = os.path.join(f'temp/dist-%s.png' % (data_name))
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


legends = ['S', 'NS']
f, ax = plt.subplots(1, 1, figsize=(8, 5))
xticklabels = ['%.2f' % dp for dp in def_prob_range]
xticks = range(len(def_prob_range))
for i, leg in enumerate(legends):
    ax.errorbar(xticks, delta_norm_mu[:, i],
                yerr=delta_norm_se[:, i], label=leg)
ax.set_title(data_name)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_ylabel('Change of activity')
ax.legend()
f.tight_layout()
sns.despine()

fig_path = os.path.join(f'temp/delta-act-%s.png' % (data_name))
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
