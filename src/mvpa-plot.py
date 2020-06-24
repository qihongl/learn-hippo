import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from vis import print_dict
from utils.io import pickle_load_dict
from analysis import compute_stats

# define some constants
sns.set(style='white', palette='colorblind', context='poster')
all_conditions = ['no schema', 'schema consistent', 'schema violated']
all_outcome = ['correct', 'mistake', 'dk']
max_response_type = ['studied', 'dk', 'other']
def_prob_range = np.arange(.3, .9, .1)
len(def_prob_range)

# prealloc
all_counts_dicts = {'%.1f' % def_prob: None for def_prob in def_prob_range}
all_pmu_dicts = {'%.1f' % def_prob: None for def_prob in def_prob_range}
all_pse_dicts = {'%.1f' % def_prob: None for def_prob in def_prob_range}
prop_dict_co_mu = {c: {o: np.zeros((len(def_prob_range), 3)) for o in all_outcome}
                   for c in all_conditions}
prop_dict_co_se = {c: {o: np.zeros((len(def_prob_range), 3)) for o in all_outcome}
                   for c in all_conditions}
prop_dict_cm_mu = {
    c: {mrt: np.zeros((len(def_prob_range), 3)) for mrt in max_response_type}
    for c in all_conditions
}
prop_dict_cm_se = {
    c: {mrt: np.zeros((len(def_prob_range), 3)) for mrt in max_response_type}
    for c in all_conditions
}
enc_acc_gmu = np.zeros(len(def_prob_range))
enc_acc_gse = np.zeros(len(def_prob_range))
p_schematic_enc_err_gmu = np.zeros(len(def_prob_range))
p_schematic_enc_err_gse = np.zeros(len(def_prob_range))
schm_v_mr_schematic = {
    '%.1f' % def_prob: [] for def_prob in def_prob_range}

# for all schema levels
for dpi, def_prob in enumerate(def_prob_range):
    print('%d -  def_prob = %.1f' % (dpi, def_prob))
    # load data
    mvpa_data_dict_fname = 'mvpa-schema-%.1f.pkl' % (def_prob)
    mvpa_data_dict = pickle_load_dict(
        os.path.join('temp', mvpa_data_dict_fname))
    enc_acc_g = np.array(mvpa_data_dict['enc_acc_g'])
    p_schematic_enc_err_g = np.array(
        mvpa_data_dict['schematic_enc_err_rate_g'])
    df_g = mvpa_data_dict['df_g']

    ''' compute encoding performance'''
    none_loc = enc_acc_g == None
    if np.sum(none_loc) > 0:
        none_loc = np.where(none_loc)[0]
        enc_acc_g = np.delete(enc_acc_g, none_loc)

    enc_acc_gmu[dpi], enc_acc_gse[dpi] = compute_stats(enc_acc_g)
    p_schematic_enc_err_g = p_schematic_enc_err_g[p_schematic_enc_err_g != None]
    p_schematic_enc_err_gmu[dpi], p_schematic_enc_err_gse[dpi] = compute_stats(
        p_schematic_enc_err_g)

    '''by condition, then by outcome, decoded response counts counts'''
    counts_scom_dicts, counts_scmo_dicts = [], []
    n_subjs = len(df_g)
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
        counts_com_dict = {condition: None for condition in all_conditions}
        counts_cmo_dict = {condition: None for condition in all_conditions}
        # loop over all schema-based condition
        for condition, df_i in zip(all_conditions, all_df):
            df_i = df_i.loc[df_i['has_enc_err'] == True]
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
                # print(dict_io)
                #
                if outcome == 'mistake' and condition == 'schema violated':
                    df_io_other = df_io.loc[df_io['max_response'] == 'other']
                    schm_v_mr_schematic['%.1f' % def_prob].extend(
                        list(df_io_other['max_response_schematic'])
                    )
                # collect the data
                outcome_counts[outcome] = list(dict_io.values())
            counts_com_dict[condition] = outcome_counts

            # get scmo count data
            mrt_counts = {mrt: 0 for mrt in max_response_type}
            for mrt in max_response_type:
                df_im = df_i.loc[df_i['max_response'] == mrt]
                vcounts = df_im['outcome'].value_counts()
                values = vcounts.keys().tolist()
                counts = vcounts.tolist()

                dict_im = {o: 0 for o in all_outcome}
                for o, count_i in zip(values, counts):
                    dict_im[o] = count_i
                mrt_counts[mrt] = list(dict_im.values())
            counts_cmo_dict[condition] = mrt_counts

        # collect the data for this subj
        counts_scom_dicts.append(counts_com_dict)
        counts_scmo_dicts.append(counts_cmo_dict)

    # compute group average for this schema level
    counts_co_dict_mu = {c: {o: None for o in all_outcome}
                         for c in all_conditions}
    counts_co_dict_se = {c: {o: None for o in all_outcome}
                         for c in all_conditions}

    n_subj_ = len(counts_scom_dicts)
    for condition in all_conditions:
        for outcome in all_outcome:
            # for i_s in range(len(counts_scom_dicts)):
            counts_io_ = np.array([counts_scom_dicts[i_s][condition][outcome]
                                   for i_s in range(n_subj_)])
            prop_io_ = counts_io_ / np.sum(counts_io_, axis=1, keepdims=True)
            mu_, se_ = compute_stats(counts_io_)
            pmu_, pse_ = compute_stats(prop_io_)

            counts_co_dict_mu[condition][outcome] = mu_
            counts_co_dict_se[condition][outcome] = se_
            if np.sum(mu_) == 0:
                prop_dict_co_mu[condition][outcome][dpi,
                                                    :] = np.array([np.nan] * 3)
                prop_dict_co_se[condition][outcome][dpi,
                                                    :] = np.array([np.nan] * 3)
            else:
                prop_dict_co_mu[condition][outcome][dpi, :] = pmu_
                prop_dict_co_se[condition][outcome][dpi, :] = pse_

        counts_cm_dict_mu = {c: {mrt: None for mrt in max_response_type}
                             for c in all_conditions}
        counts_cm_dict_se = {c: {mrt: None for mrt in max_response_type}
                             for c in all_conditions}

        for mrt in max_response_type:
            counts_io_ = np.array([counts_scmo_dicts[i_s][condition][mrt]
                                   for i_s in range(n_subj_)])
            prop_io_ = counts_io_ / np.sum(counts_io_, axis=1, keepdims=True)
            mu_, se_ = compute_stats(counts_io_)
            pmu_, pse_ = compute_stats(prop_io_)

            counts_co_dict_mu[condition][mrt] = mu_
            counts_co_dict_se[condition][mrt] = se_
            if np.sum(mu_) == 0:
                prop_dict_cm_mu[condition][mrt][dpi,
                                                :] = np.array([np.nan] * 3)
                prop_dict_cm_se[condition][mrt][dpi,
                                                :] = np.array([np.nan] * 3)
            else:
                prop_dict_cm_mu[condition][mrt][dpi, :] = pmu_
                prop_dict_cm_se[condition][mrt][dpi, :] = pse_

    # collect by schema level
    all_counts_dicts['%.1f' % def_prob] = counts_scom_dicts
#     all_pmu_dicts['%.1f' % def_prob] = prop_dict_co_mu
#     all_pse_dicts['%.1f' % def_prob] = prop_dict_co_se

# print_dict(all_pmu_dicts)
# print_dict(all_pmu_dicts)
# np.sum(all_pmu_dicts['0.3']['schema violated']['mistake'], axis=1)

'''by condition, outcome counts AND by condition, decoded response counts'''
outcome_mu = {'%.1f' % def_prob: None for def_prob in def_prob_range}
outcome_se = {'%.1f' % def_prob: None for def_prob in def_prob_range}
dr_mu = {'%.1f' % def_prob: None for def_prob in def_prob_range}
dr_se = {'%.1f' % def_prob: None for def_prob in def_prob_range}
for dpi, def_prob in enumerate(def_prob_range):
    counts_dicts_dpi = all_counts_dicts['%.1f' % def_prob]
    n_subj_dpi = len(counts_dicts_dpi)
    outcome_mu_bycond = {condition: None for condition in all_conditions}
    outcome_se_bycond = {condition: None for condition in all_conditions}
    dr_mu_bycond = {condition: None for condition in all_conditions}
    dr_se_bycond = {condition: None for condition in all_conditions}
    for ci, condition in enumerate(all_conditions):
        outcome_bysub = np.zeros((len(all_outcome), n_subj_dpi))
        dr_bysub = np.zeros((len(max_response_type), n_subj_dpi))
        for si in range(n_subj_dpi):
            for oi, outcome in enumerate(all_outcome):
                outcome_bysub[oi, si] = np.sum(
                    counts_dicts_dpi[si][condition][outcome])
            # collapse across outcomes, merge decoded responses
            dr_cond_si = np.sum(np.array([counts_dicts_dpi[si][condition][outcome]
                                          for outcome in all_outcome]), axis=0)
            dr_bysub[:, si] = dr_cond_si
        # each mu number is the average outcome count across all subjs
        # arranged by correct, mistake, dk
        outcome_porp_bysub = outcome_bysub / \
            np.sum(outcome_bysub, axis=0, keepdims=True)
        outcome_mu_bycond[condition], outcome_se_bycond[condition] = compute_stats(
            outcome_porp_bysub.T)
        dr_prop_bysub = dr_bysub / np.sum(dr_bysub, axis=0, keepdims=True)
        dr_mu_bycond[condition], dr_se_bycond[condition] = compute_stats(
            dr_prop_bysub.T)
    outcome_mu['%.1f' % def_prob] = outcome_mu_bycond
    outcome_se['%.1f' % def_prob] = outcome_se_bycond
    dr_mu['%.1f' % def_prob] = dr_mu_bycond
    dr_se['%.1f' % def_prob] = dr_se_bycond

outcome_mu_byss = {
    condition:
    [outcome_mu['%.1f' % def_prob][condition] for def_prob in def_prob_range]
    for condition in all_conditions
}
outcome_se_byss = {
    condition:
    [outcome_se['%.1f' % def_prob][condition] for def_prob in def_prob_range]
    for condition in all_conditions
}

dr_mu_byss = {
    condition:
    [dr_mu['%.1f' % def_prob][condition] for def_prob in def_prob_range]
    for condition in all_conditions
}
dr_se_byss = {
    condition:
    [dr_se['%.1f' % def_prob][condition] for def_prob in def_prob_range]
    for condition in all_conditions
}

'''by condition, decoded response counts'''


'''plotting params'''

# sns.palplot(sns.color_palette())
color_y = sns.color_palette()[1]
color_g = sns.color_palette()[2]
color_r = sns.color_palette()[3]
color_b = sns.color_palette()[0]
width = 0.6
error_kw = dict()
xticks = range(len(def_prob_range))

'''1. by condition, outcome counts'''
f, axes = plt.subplots(3, 1, figsize=(8, 12))
for ci, condition in enumerate(all_conditions):
    correct_mu_bars = np.array(outcome_mu_byss[condition])[:, 0]
    mistake_mu_bars = np.array(outcome_mu_byss[condition])[:, 1]
    dk_mu_bars = np.array(outcome_mu_byss[condition])[:, 2]
    correct_se_bars = np.array(outcome_se_byss[condition])[:, 0]
    mistake_se_bars = np.array(outcome_se_byss[condition])[:, 1]
    dk_se_bars = np.array(outcome_se_byss[condition])[:, 2]

    axes[ci].bar(xticks, correct_mu_bars, width,
                 yerr=correct_se_bars, label='correct', color=color_g,
                 error_kw=error_kw)
    axes[ci].bar(xticks, dk_mu_bars, width,
                 yerr=dk_se_bars, bottom=correct_mu_bars,
                 label='mistake', color=color_r, error_kw=error_kw)
    axes[ci].bar(xticks, mistake_mu_bars, width, yerr=mistake_se_bars,
                 bottom=correct_mu_bars + dk_mu_bars,
                 label='dk', color=color_b, error_kw=error_kw)
    axes[ci].set_title(condition)
    axes[ci].set_ylabel('%')
    axes[ci].set_xticks(xticks)
    axes[ci].set_xticklabels(['%.1f' % dp for dp in def_prob_range])
    axes[ci].set_yticks([0, .5, 1])
    axes[ci].set_ylim([-.025, 1.025])
    axes[ci].set_xlim([-.5, 6.5])
    axes[ci].set_yticklabels([0, .5, 1])
axes[-1].legend()
axes[-1].set_xlabel('Schema strength')
sns.despine()
f.tight_layout()
img_name = 'mvpa-outcome.png'
f.savefig(os.path.join('../figs', img_name))

'''2. by condition, decoded response counts'''
f, axes = plt.subplots(3, 1, figsize=(8, 12))
for ci, condition in enumerate(all_conditions):
    stud_mu_bars = np.array(dr_mu_byss[condition])[:, 0]
    dk_mu_bars = np.array(dr_mu_byss[condition])[:, 1]
    other_mu_bars = np.array(dr_mu_byss[condition])[:, 2]
    stud_se_bars = np.array(dr_se_byss[condition])[:, 0]
    dk_se_bars = np.array(dr_se_byss[condition])[:, 1]
    other_se_bars = np.array(dr_se_byss[condition])[:, 2]

    axes[ci].bar(xticks, stud_mu_bars, width,
                 yerr=stud_se_bars, label='studied', color=color_g,
                 error_kw=error_kw)
    axes[ci].bar(xticks, dk_mu_bars, width,
                 yerr=dk_se_bars, bottom=stud_mu_bars,
                 label='dk', color=color_b, error_kw=error_kw)
    axes[ci].bar(xticks, other_mu_bars, width, yerr=other_se_bars,
                 bottom=stud_mu_bars + dk_mu_bars,
                 label='other', color=color_r, error_kw=error_kw)
    axes[ci].set_title(condition)
    axes[ci].set_ylabel('%')
    axes[ci].set_xticks(xticks)
    axes[ci].set_xticklabels(['%.1f' % dp for dp in def_prob_range])
    axes[ci].set_yticks([0, .5, 1])
    axes[ci].set_ylim([-.025, 1.025])
    axes[ci].set_xlim([-.5, 6.5])
    axes[ci].set_yticklabels([0, .5, 1])
axes[-1].legend(loc=3)
axes[-1].set_xlabel('Schema strength')
sns.despine()
f.tight_layout()
img_name = 'mvpa-dr.png'
f.savefig(os.path.join('../figs', img_name))

'''3. by condition, then by outcome, decoded response counts'''
for condition in all_conditions:
    f, axes = plt.subplots(3, 1, figsize=(8, 12))
    for i, outcome in enumerate(all_outcome):
        np.any(np.isnan(prop_dict_co_mu[condition][outcome]), axis=1)

        print(prop_dict_co_mu[condition][outcome])

        # max_response_type
        mrt_stud = prop_dict_co_mu[condition][outcome][:, 0]
        mrt_dk = prop_dict_co_mu[condition][outcome][:, 1]
        mrt_other = prop_dict_co_mu[condition][outcome][:, 2]
        mrt_stud_se = prop_dict_co_se[condition][outcome][:, 0]
        mrt_dk_se = prop_dict_co_se[condition][outcome][:, 1]
        mrt_other_se = prop_dict_co_se[condition][outcome][:, 2]

        axes[i].bar(xticks, mrt_stud, width, yerr=mrt_stud_se,
                    label='studied', color=color_g, error_kw=error_kw)
        axes[i].bar(xticks, mrt_dk, width, yerr=mrt_dk_se,
                    bottom=mrt_stud, label='dk', color=color_b, error_kw=error_kw)
        axes[i].bar(xticks, mrt_other, width, yerr=mrt_other_se,
                    bottom=mrt_stud + mrt_dk, label='other', color=color_r,
                    error_kw=error_kw)

        if i == 0:
            axes[i].set_title(f'Condition = {condition}\noutcome = {outcome}')
        else:
            axes[i].set_title(f'outcome = {outcome}')
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

    img_name = 'mvpa-dr-counts-%s.png' % (condition)
    f.savefig(os.path.join('../figs', img_name))


'''4. by condition, then by mrt, outcome counts'''
for condition in all_conditions:
    f, axes = plt.subplots(3, 1, figsize=(8, 12))
    for i, mrt in enumerate(max_response_type):
        # max_response_type
        mrt_stud = prop_dict_cm_mu[condition][mrt][:, 0]
        mrt_dk = prop_dict_cm_mu[condition][mrt][:, 1]
        mrt_other = prop_dict_cm_mu[condition][mrt][:, 2]
        mrt_stud_se = prop_dict_cm_se[condition][mrt][:, 0]
        mrt_dk_se = prop_dict_cm_se[condition][mrt][:, 1]
        mrt_other_se = prop_dict_cm_se[condition][mrt][:, 2]

        axes[i].bar(xticks, mrt_stud, width, yerr=mrt_stud_se,
                    label='correct', color=color_g, error_kw=error_kw)
        axes[i].bar(xticks, mrt_dk, width, yerr=mrt_dk_se,
                    bottom=mrt_stud, label='mistake', color=color_r, error_kw=error_kw)
        axes[i].bar(xticks, mrt_other, width, yerr=mrt_other_se,
                    bottom=mrt_stud + mrt_dk, label='dk', color=color_b,
                    error_kw=error_kw)

        if i == 0:
            axes[i].set_title(f'Condition = {condition}\ndecoded = {mrt}')
        else:
            axes[i].set_title(f'decoded = {mrt}')
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

    img_name = 'mvpa-outcome-counts-%s.png' % (condition)
    f.savefig(os.path.join('../figs', img_name))

'''encoding error'''
f, axes = plt.subplots(2, 1, figsize=(8, 10))
axes[0].bar(xticks, enc_acc_gmu, width, yerr=enc_acc_gse)
axes[0].axhline(1, linestyle='--', color='grey')
# axes[0].set_xlabel('Schema strength')
axes[0].set_ylabel('Encoding accuracy')
axes[0].set_yticks([.6, .8, 1])
axes[0].set_ylim([.6, 1.05])

axes[1].bar(xticks, p_schematic_enc_err_gmu,
            width, yerr=p_schematic_enc_err_gse, color=color_r)
axes[1].set_xlabel('Schema strength')
axes[1].set_ylabel('% encoding error schematic')
axes[1].set_yticks([.0, .5, 1])

for ax in axes:
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%.1f' % dp for dp in def_prob_range])
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
