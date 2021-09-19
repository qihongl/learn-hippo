import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from task import SimplePairedAssociate2
from utils.utils import to_sqnp, batch_sqnp, to_np, to_pth
from utils.io import pickle_load_dict
from analysis import compute_stats
from analysis.spa import *


sns.set(style='white', palette='colorblind', context='talk')
cpals = sns.color_palette()
log_root = '../log/'
sim_name = 'cong'
selective_encoding = False

'''init task and model'''
cmpt = .8
# for cmpt in [0, .2, .4, .6, .8, 1]:
n_epochs = 600
n_cue = 16
n_assoc = 32
schema_level = .5
spa = SimplePairedAssociate2(
    n_cue=n_cue, n_assoc=n_assoc, schema_level=schema_level
)

len_study_phase = spa.n_cue * 2
T = spa.n_cue * 3
rnn_hidden_dim = 32
dec_hidden_dim = 16

lr = 3e-3
pe_threshold = .03

# subj_id = 1
n_subjs = 10
acc = np.zeros((2, n_subjs))
inpt = np.zeros((2, n_subjs))
mem_act = np.zeros((4, n_subjs))
mem_cos = np.zeros((4, n_subjs))

for subj_id in range(n_subjs):
    print(subj_id)
    np.random.seed(subj_id)

    # create dir
    log_dir = os.path.join(
        log_root, 'schema-%.2f/cmpt-%.2f' % (schema_level, cmpt))
    # create data name
    enc_pol = 1 if selective_encoding else 0
    data_path = os.path.join(log_dir, f'subj-{subj_id}-enc-{enc_pol}.pkl')
    if not os.path.exists(data_path):
        raise ValueError(f'DATA NOT FOUND: {data_path}')

    data_dict = pickle_load_dict(data_path)

    # pack data
    Log_loss = data_dict['Log_loss']
    Log_inpt = data_dict['Log_inpt']
    Log_acc = data_dict['Log_acc']
    Log_cond = data_dict['Log_cond']
    mem_act_list = data_dict['mem_act_list']
    mem_cos_list = data_dict['mem_cos_list']

    # compute accuracy for cong, incong
    acc_cong_tst, acc_incong_tst = sep_data(
        Log_acc[:, n_cue:], Log_cond[:, 1, :]
    )
    acc[0, subj_id] = np.mean(acc_cong_tst)
    acc[1, subj_id] = np.mean(acc_incong_tst)

    # collect memory activaiton and cosine similarity
    # ta_cong_mu, la_cong_mu, ta_incong_mu, la_incong_mu
    mem_act[:, subj_id] = mem_act_list
    mem_cos[:, subj_id] = mem_cos_list

    # input gate
    inpt_cng, inpt_icg = sep_data(
        Log_inpt[:, n_cue * 2:], Log_cond[:, 1, :])
    inpt[:, subj_id] = [np.mean(inpt_cng), np.mean(inpt_icg)]

# compute group level stats for acc and mem act
acc_mu, acc_se = compute_stats(acc, axis=1)
inpt_mu, inpt_se = compute_stats(inpt, axis=1)
mem_act_mu, mem_act_se = compute_stats(mem_act, axis=1)
mem_cos_mu, mem_cos_se = compute_stats(mem_cos, axis=1)

cpal = sns.color_palette()
f, ax = plt.subplots(1, 1, figsize=(4, 4))
lgds = ['congruent', 'incongruent']
ax.bar(x=range(2), height=acc_mu, yerr=acc_se, color=cpal[:2])
ax.set_ylim([0, 1])
ax.set_ylabel('Accuracy')
ax.set_xticks(range(2))
ax.set_xticklabels(lgds)
ax.set_xlabel(' ')
f.tight_layout()
sns.despine()
f.savefig(f'../figs/acc-cmpt-{cmpt}-encpol-{enc_pol}.png', dpi=100)

f, ax = plt.subplots(1, 1, figsize=(4, 4))
lgds = ['congruent', 'incongruent']
ax.bar(x=range(2), height=inpt_mu, yerr=inpt_se, color=cpal[:2])
ax.set_ylim([0, 1])
ax.set_ylabel('EM gate')
ax.set_xticks(range(2))
ax.set_xticklabels(lgds)
f.tight_layout()
sns.despine()
f.savefig(f'../figs/inpt-cmpt-{cmpt}-encpol-{enc_pol}.png', dpi=100)

f, ax = plt.subplots(1, 1, figsize=(2.5, 4))
ax.bar(x=range(1), height=np.mean(inpt_mu), yerr=np.mean(inpt_se))
ax.set_ylim([0, 1])
ax.set_ylabel('EM gate')
ax.set_xticks(range(1))
ax.set_xticklabels(' ')
ax.set_xlabel(' ')
f.tight_layout()
sns.despine()
f.savefig(f'../figs/minpt-cmpt-{cmpt}-encpol-{enc_pol}.png', dpi=100)

gr_pal = sns.color_palette()[2:4]
f, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
mem_act_mu[np.isnan(mem_act_mu)] = 0
axes[0].bar(x=range(2), height=mem_act_mu[:2],
            yerr=mem_act_se[:2], color=gr_pal)
axes[0].set_ylabel('Memory activation')
axes[0].set_xticks(range(2))
axes[0].set_xticklabels(['targ', 'lure'])
axes[0].set_xlabel('congruent')

axes[1].bar(x=range(2), height=mem_act_mu[2:],
            yerr=mem_act_se[2:], color=gr_pal)
axes[1].set_ylabel('Memory activation')
axes[1].set_xticks(range(2))
axes[1].set_xticklabels(['targ', 'lure'])
axes[1].set_xlabel('incongruent')
axes[1].set_ylim([0, 1])
f.tight_layout()
sns.despine()
f.savefig(f'../figs/ma-cmpt-{cmpt}-encpol-{enc_pol}.png', dpi=100)

gr_pal = sns.color_palette()[2:4]
f, ax = plt.subplots(1, 1, figsize=(4, 4), sharey=True)
ax.bar(x=range(2), height=mem_act_mu[:2],
       yerr=mem_act_se[:2], color=gr_pal)
ax.set_ylabel('Memory activation')
ax.set_xticks(range(2))
ax.set_xticklabels(['targ', 'lure'])
ax.set_xlabel('congruent')
ax.set_ylim([0, 1])
f.tight_layout()
sns.despine()
f.savefig(f'../figs/ma-cmpt-{cmpt}-encpol-{enc_pol}-cong.png', dpi=100)

f, ax = plt.subplots(1, 1, figsize=(4, 4), sharey=True)
ax.bar(x=range(2), height=mem_act_mu[2:],
       yerr=mem_act_se[2:], color=gr_pal)
ax.set_ylabel('Memory activation')
ax.set_xticks(range(2))
ax.set_xticklabels(['targ', 'lure'])
ax.set_xlabel('incongruent')
ax.set_ylim([0, 1])
f.tight_layout()
sns.despine()
f.savefig(f'../figs/ma-cmpt-{cmpt}-encpol-{enc_pol}-incong.png', dpi=100)
