import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from vis import print_dict
from utils.io import pickle_load_dict
from analysis import compute_stats

sns.set(style='white', palette='colorblind', context='poster')
all_conditions = ['no schema', 'schema consistent', 'schema violated']
all_outcome = ['correct', 'mistake', 'dk']
max_response_type = ['studied', 'dk', 'other']

def_prob_range = np.arange(.3, .9, .1)


all_pmu_dicts = {'%.1f' % def_prob: None for def_prob in def_prob_range}
all_pse_dicts = {'%.1f' % def_prob: None for def_prob in def_prob_range}
prop_dict_mu = {c: {o: np.zeros((len(def_prob_range), 3)) for o in all_outcome}
                for c in all_conditions}
prop_dict_se = {c: {o: np.zeros((len(def_prob_range), 3)) for o in all_outcome}
                for c in all_conditions}
enc_acc_gmu = np.zeros(len(def_prob_range))
enc_acc_gse = np.zeros(len(def_prob_range))
schm_v_mr_schematic = {
    '%.1f' % def_prob: [] for def_prob in def_prob_range}

for dpi, def_prob in enumerate(def_prob_range):
    print('%d -  def_prob = %.1f' % (dpi, def_prob))
    # load data
    mvpa_data_dict_fname = 'mvpa-schema-%.1f.pkl' % (def_prob)
    mvpa_data_dict = pickle_load_dict(
        os.path.join('temp', mvpa_data_dict_fname))
    enc_acc_g = np.array(mvpa_data_dict['enc_acc_g'])
    df_g = mvpa_data_dict['df_g']

    '''encoding performance'''
    none_loc = enc_acc_g == None
    if np.sum(none_loc) > 0:
        none_loc = np.where(none_loc)[0]
        enc_acc_g = np.delete(enc_acc_g, none_loc)

    enc_acc_gmu[dpi], enc_acc_gse[dpi] = compute_stats(enc_acc_g)

    '''detailed performance analysis'''
    # i_s = 1
    # df_g[9]
    # df_i = df_is_sic
    # outcome = 'mistake'

    counts_dicts = []
    n_subjs = len(df_g)
    # schm_v_mr_schematic = []
    for i_s in range(n_subjs):
        if df_g[i_s] is None:
            continue

        # split the df w.r.t. schema condition
        df_is_sc = df_g[i_s].loc[df_g[i_s]['schema_consistent'] == True]
        df_is_sic = df_g[i_s].loc[df_g[i_s]['schema_consistent'] == False]
        df_is_ns = df_g[i_s].loc[np.isnan(
            list(df_g[i_s]['schema_consistent']))]
        all_df = [df_is_ns, df_is_sc, df_is_sic]

        # prealloc
        counts_dict = {condition: None for condition in all_conditions}
        # loop over all schema-based condition
        for condition, df_i in zip(all_conditions, all_df):
            # prealloc
            outcome_counts = {o: 0 for o in all_outcome}
            for outcome in all_outcome:
                # extract df for a specific outcome
                df_io = df_i.loc[df_i['outcome'] == outcome]
                # get the max responses
                vcounts = df_io['max_response'].value_counts()
                values = vcounts.keys().tolist()
                counts = vcounts.tolist()
                # form a response_type -> count dict
                dict_io = {mrt: 0 for mrt in max_response_type}
                for mrt, count_i in zip(values, counts):
                    dict_io[mrt] = count_i
                print(dict_io)
                #
                if outcome == 'mistake' and condition == 'schema violated':
                    df_io_other = df_io.loc[df_io['max_response'] == 'other']
                    schm_v_mr_schematic['%.1f' % def_prob].extend(
                        list(df_io_other['max_response_schematic'])
                    )
                # collect the data
                outcome_counts[outcome] = list(dict_io.values())
            counts_dict[condition] = outcome_counts
        # print the data for each condition
        counts_dicts.append(counts_dict)
        # print(schm_v_mr_schematic)
        # print_dict(counts_dict)

    # compute group average
    counts_dict_mu = {c: {o: None for o in all_outcome}
                      for c in all_conditions}
    counts_dict_se = {c: {o: None for o in all_outcome}
                      for c in all_conditions}

    for condition, df_i in zip(all_conditions, all_df):
        for outcome in all_outcome:
            # for i_s in range(len(counts_dicts)):
            counts_io_ = np.array([counts_dicts[i_s][condition][outcome]
                                   for i_s in range(len(counts_dicts))])
            prop_io_ = counts_io_ / np.sum(counts_io_, axis=1, keepdims=True)
            mu_, se_ = compute_stats(counts_io_)
            pmu_, pse_ = compute_stats(prop_io_)

            counts_dict_mu[condition][outcome] = mu_
            counts_dict_se[condition][outcome] = se_
            if np.sum(mu_) == 0:
                prop_dict_mu[condition][outcome][dpi,
                                                 :] = np.array([np.nan] * 3)
                prop_dict_se[condition][outcome][dpi,
                                                 :] = np.array([np.nan] * 3)
            else:
                prop_dict_mu[condition][outcome][dpi, :] = pmu_
                prop_dict_se[condition][outcome][dpi, :] = pse_

    all_pmu_dicts['%.1f' % def_prob] = prop_dict_mu
    all_pse_dicts['%.1f' % def_prob] = prop_dict_se

print_dict(all_pmu_dicts)
print_dict(all_pmu_dicts)
np.sum(all_pmu_dicts['0.3']['schema violated']['mistake'], axis=1)


'''plot count data'''
# sns.palplot(sns.color_palette())
color_g = sns.color_palette()[2]
color_r = sns.color_palette()[3]
color_b = sns.color_palette()[0]

width = 0.5
error_kw = dict()
xticks = range(len(def_prob_range))
for condition, df_i in zip(all_conditions, all_df):
    f, axes = plt.subplots(3, 1, figsize=(8, 12))
    for i, outcome in enumerate(all_outcome):
        np.any(np.isnan(prop_dict_mu[condition][outcome]), axis=1)

        print(prop_dict_mu[condition][outcome])

        # max_response_type
        mrt_stud = prop_dict_mu[condition][outcome][:, 0]
        mrt_dk = prop_dict_mu[condition][outcome][:, 1]
        mrt_other = prop_dict_mu[condition][outcome][:, 2]
        mrt_stud_se = prop_dict_se[condition][outcome][:, 0]
        mrt_dk_se = prop_dict_se[condition][outcome][:, 1]
        mrt_other_se = prop_dict_se[condition][outcome][:, 2]

        axes[i].bar(xticks, mrt_stud, width, yerr=mrt_stud_se,
                    label='studied', color=color_g, error_kw=error_kw)
        axes[i].bar(xticks, mrt_dk, width, yerr=mrt_dk_se,
                    bottom=mrt_stud, label='dk', color=color_b, error_kw=error_kw)
        axes[i].bar(xticks, mrt_other, width, yerr=mrt_other_se,
                    bottom=mrt_stud + mrt_dk, label='other', color=color_r,
                    error_kw=error_kw)

        if i == 0:
            axes[i].set_title(f'Trial type = {condition}\noutcome = {outcome}')
        else:
            axes[i].set_title(f'outcome = {outcome}')
        axes[i].set_ylabel('Schema strength')
        axes[i].set_ylabel('%')
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels(['%.1f' % dp for dp in def_prob_range])
        axes[i].set_yticks([0, .5, 1])
        axes[i].set_ylim([-.025, 1.025])
        axes[i].set_xlim([-.5, 6.5])
        axes[i].set_yticklabels([0, .5, 1])
    axes[-1].legend()
    axes[-1].set_xlabel('Schema strength')
    sns.despine()
    f.tight_layout()

    img_name = 'mvpa-count-%s.png' % (condition)
    f.savefig(os.path.join('../figs', img_name))

'''encoding error'''
f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.bar(xticks, enc_acc_gmu, width, yerr=enc_acc_gse)
ax.axhline(1, linestyle='--', color='grey')
ax.set_xlabel('Schema strength')
ax.set_ylabel('Encoding accuracy')
ax.set_xticks(xticks)
ax.set_xticklabels(['%.1f' % dp for dp in def_prob_range])
ax.set_yticks([.6, .8, 1])
ax.set_ylim([.6, 1.05])
ax.set_xlim([-.5, 6.5])
sns.despine()
f.tight_layout()

img_name = 'enc-err.png'
f.savefig(os.path.join('../figs', img_name))


'''schema consistent error'''
mr_schematic_rate = np.zeros(len(def_prob_range))
for dpi, def_prob in enumerate(def_prob_range):
    arr_ = np.array(schm_v_mr_schematic['%.1f' % def_prob])
    mr_schematic_rate[dpi] = np.sum(arr_) / len(arr_)

'''MR schema consistent in schema vio trials'''
f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.bar(xticks, mr_schematic_rate, width)
ax.set_xlabel('Schema strength')
ax.set_ylabel('%')
ax.set_title(
    'Average proportion schematic response \nfor all schema violation trials')
ax.set_xticks(xticks)
ax.set_xticklabels(['%.1f' % dp for dp in def_prob_range])
ax.set_yticks([0, .25])
ax.set_ylim([-.01, .35])
ax.set_xlim([-.5, 6.5])
sns.despine()
f.tight_layout()
img_name = 'prop-other-sch-con.png'
f.savefig(os.path.join('../figs', img_name))
