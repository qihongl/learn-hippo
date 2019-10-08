import numpy as np
from utils.io import pickle_load_dict
from analysis import compute_acc, compute_dk
from vis import plot_pred_acc_full
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set(style='white', palette='colorblind', context='poster')

data16 = pickle_load_dict('temp/enc16.pkl')
data8 = pickle_load_dict('temp/enc8.pkl')


Y16 = data16['Y']
dist_a16 = data16['dist_a']
cond_ids16 = data16['cond_ids']

Y8 = data8['Y']
dist_a8 = data8['dist_a']
cond_ids8 = data8['cond_ids']

all_conds = ['RM', 'DM', 'NM']
for i, cn in enumerate(all_conds):

    f, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    Y_ = Y16[cond_ids16[cn], :]
    dist_a_ = dist_a16[cond_ids16[cn], :]
    # compute performance for this condition
    acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
    dk_mu = compute_dk(dist_a_)
    T_part = 16
    ax.errorbar(x=range(T_part*2), y=acc_mu, yerr=acc_er,
                label='encode at event boundary')

    Y_ = Y8[cond_ids8[cn], :]
    dist_a_ = dist_a8[cond_ids8[cn], :]
    # compute performance for this condition
    acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
    dk_mu = compute_dk(dist_a_)

    ax.errorbar(x=range(T_part*2), y=acc_mu, yerr=acc_er,
                label='also encode within an event')

    ax.axvline(16, color='red', linestyle='--', alpha=.5)
    ax.set_title(f'{cn}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Accuracy')
    sns.despine()
    f.tight_layout()
    ax.set_ylim([-.05, 1.05])
    # ax.legend()
    fig_path = os.path.join('temp', f'tz-acc-{cn}.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')


f, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, cn in enumerate(all_conds):

    Y_ = Y16[cond_ids16[cn], :]
    dist_a_ = dist_a16[cond_ids16[cn], :]
    # compute performance for this condition
    acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
    dk_mu = compute_dk(dist_a_)
    T_part = 16
    axes[i].errorbar(x=range(T_part), y=acc_mu[T_part:], yerr=acc_er[T_part:],
                     label='encode at event boundary')

    Y_ = Y8[cond_ids8[cn], :]
    dist_a_ = dist_a8[cond_ids8[cn], :]
    # compute performance for this condition
    acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
    dk_mu = compute_dk(dist_a_)

    axes[i].errorbar(x=range(T_part), y=acc_mu[T_part:], yerr=acc_er[T_part:],
                     label='also encode within an event')

    axes[i].set_title(f'{cn}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Accuracy')
    axes[i].set_ylim([-.05, 1.05])
    axes[i].set_xticks([0, 15])
    sns.despine()
    f.tight_layout()
# axes[0].legend()
fig_path = os.path.join('temp', f'tz-acc-horizontal.png')
f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')