import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from utils.io import pickle_load_dict
from analysis import compute_stats, remove_none

# define some constants
sns.set(style='white', palette='colorblind', context='poster')
all_conditions = ['no schema', 'schema consistent', 'schema violated']
# all_outcome = ['correct', 'dk', 'mistake']
# max_response_type = ['studied', 'dk', 'other']
all_outcome = ['correct', 'dk', 'mistake', 'mistake-S']
max_response_type = ['studied', 'dk', 'other', 'other-S']
def_prob_range = np.arange(.25, 1, .1)


def split_df_wrt_schema_condition(df):
    '''split the df w.r.t. schema condition'''
    df_is_sc = df.loc[df['schema_consistent'] == True]
    df_is_sic = df.loc[df['schema_consistent'] == False]
    df_is_ns = df.loc[np.isnan(list(df['schema_consistent']))]
    return [df_is_ns, df_is_sc, df_is_sic]


# prealloc
mvpa_acc_mu = {'%.2f' % dp: None for dp in def_prob_range}
mvpa_acc_se = {'%.2f' % dp: None for dp in def_prob_range}
all_enc_dicts = {'%.2f' % dp: None for dp in def_prob_range}
all_scom_counts_dicts = {'%.2f' % dp: None for dp in def_prob_range}
all_scmo_counts_dicts = {'%.2f' % dp: None for dp in def_prob_range}
all_pmu_dicts = {'%.2f' % dp: None for dp in def_prob_range}
all_pse_dicts = {'%.2f' % dp: None for dp in def_prob_range}
prop_dict_co_mu = {c: {o: np.zeros((len(def_prob_range), 4)) for o in all_outcome}
                   for c in all_conditions}
prop_dict_co_se = {c: {o: np.zeros((len(def_prob_range), 4)) for o in all_outcome}
                   for c in all_conditions}
prop_dict_cm_mu = {
    c: {mrt: np.zeros((len(def_prob_range), 4)) for mrt in max_response_type}
    for c in all_conditions
}
prop_dict_cm_se = {
    c: {mrt: np.zeros((len(def_prob_range), 4)) for mrt in max_response_type}
    for c in all_conditions
}
enc_acc_gmu = np.zeros(len(def_prob_range))
enc_acc_gse = np.zeros(len(def_prob_range))
p_schematic_enc_err_gmu = np.zeros(len(def_prob_range))
p_schematic_enc_err_gse = np.zeros(len(def_prob_range))

# for all schema levels
# dpi, def_prob = 6, .7
# exp_name = '0717-dp'
# exp_name = '0916-widesim-pfixed'
exp_name = '1029-schema-evenodd-pfixed'
# def_prob_range = np.arange(.25, 1, .1)

for dpi, def_prob in enumerate(def_prob_range):
    # for dpi, def_prob in enumerate(def_prob_range):
    print('%d -  def_prob = %.2f' % (dpi, def_prob))
    # load data
    mvpa_data_dict_fname = 'new-mvpa-schema-%.2f.pkl' % (def_prob)
    mvpa_data_dict = pickle_load_dict(
        os.path.join('temp', mvpa_data_dict_fname))
    enc_acc_g = np.array(mvpa_data_dict['enc_acc_g'])
    p_schematic_enc_err_g = np.array(
        mvpa_data_dict['schematic_enc_err_rate_g'])
    dfs_grcl = mvpa_data_dict['df_grcl']
    dfs_genc = mvpa_data_dict['df_genc']
    mvpa_acc_g = mvpa_data_dict['match_rate_g']
    n_subjs = len(dfs_grcl)

    '''compute mvpa accuracy'''
    mvpa_acc_mu['%.2f' % def_prob], mvpa_acc_se['%.2f' % def_prob] = compute_stats(
        remove_none(np.array(mvpa_acc_g)))

    ''' compute encoding performance'''
    none_loc = enc_acc_g == None
    if np.sum(none_loc) > 0:
        none_loc = np.where(none_loc)[0]
        enc_acc_g = np.delete(enc_acc_g, none_loc)

    enc_acc_gmu[dpi], enc_acc_gse[dpi] = compute_stats(enc_acc_g)
    p_schematic_enc_err_g = p_schematic_enc_err_g[p_schematic_enc_err_g != None]

    #
    enc_dict = {c: [] for c in all_conditions}
    for i_s in range(n_subjs):
        if dfs_genc[i_s] is None:
            continue
        all_df = split_df_wrt_schema_condition(dfs_genc[i_s])

        for condition, df_sc in zip(all_conditions, all_df):
            temp_enc_dict = {c: 0 for c in max_response_type}
            vcounts = df_sc['max_response'].value_counts()
            for k, v in dict(vcounts).items():
                temp_enc_dict[k] = v
            enc_dict[condition].append(deepcopy(temp_enc_dict))
    # print_dict(enc_dict)
    all_enc_dicts['%.2f' % def_prob] = enc_dict

    '''by condition, then by outcome, decoded response counts counts'''
    counts_scom_dicts, counts_scmo_dicts = [], []
    for i_s in range(n_subjs):
        if dfs_grcl[i_s] is None:
            continue

        # split the df w.r.t. schema condition
        all_df = split_df_wrt_schema_condition(dfs_grcl[i_s])

        # outcome = 'mistake'
        # prealloc
        counts_com_dict = {condition: None for condition in all_conditions}
        counts_cmo_dict = {condition: None for condition in all_conditions}
        # loop over all schema-based condition
        # outcome='correct',condition='no schema',df_sc=df_is_ns
        for condition, df_sc in zip(all_conditions, all_df):
            # df_sc = df_sc.loc[df_sc['has_enc_err'] == True]
            # prealloc
            outcome_counts = {o: 0 for o in all_outcome}

            for outcome in all_outcome:
                # extract df for a specific outcome
                df_sco = df_sc.loc[df_sc['outcome'] == outcome]
                # get the max responses
                vcounts = df_sco['max_response'].value_counts()
                values = vcounts.keys().tolist()
                counts = vcounts.tolist()

                # form a response_type -> count dict
                dict_io = {mrt: 0 for mrt in max_response_type}
                for mrt, count_i in zip(values, counts):
                    dict_io[mrt] = count_i

                # collect the data
                outcome_counts[outcome] = list(dict_io.values())
            counts_com_dict[condition] = outcome_counts

            # get scmo count data
            mrt_counts = {mrt: 0 for mrt in max_response_type}
            for mrt in max_response_type:
                df_scm = df_sc.loc[df_sc['max_response'] == mrt]
                vcounts = df_scm['outcome'].value_counts()
                values = vcounts.keys().tolist()
                counts = vcounts.tolist()

                dict_scm = {o: 0 for o in all_outcome}
                for o, count_i in zip(values, counts):
                    dict_scm[o] = count_i

                mrt_counts[mrt] = list(dict_scm.values())

            counts_cmo_dict[condition] = mrt_counts

        # collect the data for this subj
        counts_scom_dicts.append(counts_com_dict)
        counts_scmo_dicts.append(counts_cmo_dict)

    # compute group average for this schema level
    n_subj_ = len(counts_scom_dicts)
    for condition in all_conditions:
        for outcome in all_outcome:
            counts_io_ = np.array([counts_scom_dicts[i_s][condition][outcome]
                                   for i_s in range(n_subj_)])

            is_all_zero_rows = np.all(counts_io_ == 0, axis=1)
            loc_all_zero_rows = np.where(is_all_zero_rows)[0]
            if len(loc_all_zero_rows) > 0:
                counts_io_ = np.delete(counts_io_, loc_all_zero_rows, axis=0)
            prop_io_ = counts_io_ / np.sum(counts_io_, axis=1, keepdims=True)
            pmu_, pse_ = compute_stats(prop_io_)

            prop_dict_co_mu[condition][outcome][dpi, :] = pmu_
            prop_dict_co_se[condition][outcome][dpi, :] = pse_

        for mrt in max_response_type:
            counts_io_ = np.array([counts_scmo_dicts[i_s][condition][mrt]
                                   for i_s in range(n_subj_)])
            is_all_zero_rows = np.all(counts_io_ == 0, axis=1)
            loc_all_zero_rows = np.where(is_all_zero_rows)[0]
            if len(loc_all_zero_rows) > 0:
                counts_io_ = np.delete(counts_io_, loc_all_zero_rows, axis=0)
            prop_io_ = counts_io_ / np.sum(counts_io_, axis=1, keepdims=True)
            pmu_, pse_ = compute_stats(prop_io_)

            prop_dict_cm_mu[condition][mrt][dpi, :] = pmu_
            prop_dict_cm_se[condition][mrt][dpi, :] = pse_

    # collect by schema level
    all_scom_counts_dicts['%.2f' % def_prob] = counts_scom_dicts
    all_scmo_counts_dicts['%.2f' % def_prob] = counts_scmo_dicts


'''by condition, outcome counts AND by condition, decoded response counts'''
outcome_mu = {'%.2f' % def_prob: None for def_prob in def_prob_range}
outcome_se = {'%.2f' % def_prob: None for def_prob in def_prob_range}
dr_mu = {'%.2f' % def_prob: None for def_prob in def_prob_range}
dr_se = {'%.2f' % def_prob: None for def_prob in def_prob_range}
# all_outcome_ext = all_outcome + ['mistake-schema']
for dpi, def_prob in enumerate(def_prob_range):
    scom_counts_dicts_dpi = all_scom_counts_dicts['%.2f' % def_prob]
    scmo_counts_dicts_dpi = all_scmo_counts_dicts['%.2f' % def_prob]

    n_subj_dpi = len(scom_counts_dicts_dpi)
    outcome_mu_bycond = {condition: None for condition in all_conditions}
    outcome_se_bycond = {condition: None for condition in all_conditions}
    dr_mu_bycond = {condition: None for condition in all_conditions}
    dr_se_bycond = {condition: None for condition in all_conditions}

    for ci, condition in enumerate(all_conditions):
        outcome_bysub = np.zeros((len(all_outcome), n_subj_dpi))
        dr_bysub = np.zeros((len(max_response_type), n_subj_dpi))
        outcome_bysub = np.zeros((len(all_outcome), n_subj_dpi))

        for si in range(n_subj_dpi):

            outcome_cond_si = np.sum(np.array([scmo_counts_dicts_dpi[si][condition][mrt]
                                               for mrt in max_response_type]), axis=0)
            outcome_bysub[:, si] = outcome_cond_si
            # collapse across outcomes, merge decoded responses
            dr_cond_si = np.sum(np.array([scom_counts_dicts_dpi[si][condition][outcome]
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
    outcome_mu['%.2f' % def_prob] = outcome_mu_bycond
    outcome_se['%.2f' % def_prob] = outcome_se_bycond
    dr_mu['%.2f' % def_prob] = dr_mu_bycond
    dr_se['%.2f' % def_prob] = dr_se_bycond

outcome_mu_byss = {
    condition:
    [outcome_mu['%.2f' % def_prob][condition] for def_prob in def_prob_range]
    for condition in all_conditions
}
outcome_se_byss = {
    condition:
    [outcome_se['%.2f' % def_prob][condition] for def_prob in def_prob_range]
    for condition in all_conditions
}

dr_mu_byss = {
    condition:
    [dr_mu['%.2f' % def_prob][condition] for def_prob in def_prob_range]
    for condition in all_conditions
}
dr_se_byss = {
    condition:
    [dr_se['%.2f' % def_prob][condition] for def_prob in def_prob_range]
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
xticklabels = ['%.2f' % dp for dp in def_prob_range]


'''0. mvpa acc '''
f, ax = plt.subplots(1, 1, figsize=(7, 4))
mvpa_acc_mus = np.array([v for k, v in mvpa_acc_mu.items()])
mvpa_acc_ses = np.array([v for k, v in mvpa_acc_se.items()])
for pi in range(np.shape(mvpa_acc_mus)[1]):
    ax.errorbar(x=xticks, y=mvpa_acc_mus[:, pi],
                # yerr=mvpa_acc_ses[:, pi]
                )
ax.set_ylim([0, 1])
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Schema strength')
sns.despine()
overall_mean_mvpa_acc = np.mean(mvpa_acc_mus, axis=0)
print(f'overall_mean_mvpa_acc = {overall_mean_mvpa_acc}')

'''1. by condition, outcome counts'''
f, axes = plt.subplots(3, 1, figsize=(8, 13))
for ci, condition in enumerate(all_conditions):
    correct_mu_bars = np.array(outcome_mu_byss[condition])[:, 0]
    dk_mu_bars = np.array(outcome_mu_byss[condition])[:, 1]
    mistake_mu_bars = np.array(outcome_mu_byss[condition])[:, 2]
    mistake_sc_mu_bars = np.array(outcome_mu_byss[condition])[:, 3]
    correct_se_bars = np.array(outcome_se_byss[condition])[:, 0]
    dk_se_bars = np.array(outcome_se_byss[condition])[:, 1]
    mistake_se_bars = np.array(outcome_se_byss[condition])[:, 2]
    mistake_sc_se_bars = np.array(outcome_se_byss[condition])[:, 3]

    axes[ci].bar(xticks, correct_mu_bars, width,
                 yerr=correct_se_bars, label='correct', color=color_g,
                 error_kw=error_kw)
    axes[ci].bar(xticks, dk_mu_bars, width,
                 yerr=dk_se_bars, bottom=correct_mu_bars,
                 label='don\'t know', color=color_b, error_kw=error_kw)
    axes[ci].bar(xticks, mistake_mu_bars, width, yerr=mistake_se_bars,
                 bottom=correct_mu_bars + dk_mu_bars,
                 label='mistake-non-schematic', color=color_r, error_kw=error_kw)
    axes[ci].bar(xticks, mistake_sc_mu_bars, width, yerr=mistake_sc_se_bars,
                 bottom=correct_mu_bars + dk_mu_bars + mistake_mu_bars,
                 label='mistake-schematic', color=color_y, error_kw=error_kw)
    if ci == 0:
        axes[ci].set_title('Prediction outcome, part 2\n\n%s' % condition)
    else:
        axes[ci].set_title(condition)
    axes[ci].set_ylabel('%')
    axes[ci].set_xticks(xticks)
    axes[ci].set_xticklabels(xticklabels)
    axes[ci].set_yticks([0, .5, 1])
    axes[ci].set_ylim([-.025, 1.025])
    axes[ci].set_xlim([-.5, 7.5])
    axes[ci].set_yticklabels([0, .5, 1])
# axes[-1].legend()
axes[0].legend(bbox_to_anchor=(0, 2, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=-1, ncol=2)
axes[-1].set_xlabel('Schema strength')
sns.despine()
f.tight_layout()
img_name = 'mvpa-outcome.png'
f.savefig(os.path.join('../figs', img_name), bbox_inches='tight')

'''2. by condition, decoded response counts'''
f, axes = plt.subplots(3, 1, figsize=(8, 13))
for ci, condition in enumerate(all_conditions):
    stud_mu_bars = np.array(dr_mu_byss[condition])[:, 0]
    dk_mu_bars = np.array(dr_mu_byss[condition])[:, 1]
    other_mu_bars = np.array(dr_mu_byss[condition])[:, 2]
    other_sc_mu_bars = np.array(dr_mu_byss[condition])[:, 3]
    stud_se_bars = np.array(dr_se_byss[condition])[:, 0]
    dk_se_bars = np.array(dr_se_byss[condition])[:, 1]
    other_se_bars = np.array(dr_se_byss[condition])[:, 2]
    other_sc_se_bars = np.array(dr_se_byss[condition])[:, 3]

    axes[ci].bar(xticks, stud_mu_bars, width,
                 yerr=stud_se_bars, label='studied', color=color_g,
                 error_kw=error_kw)
    axes[ci].bar(xticks, dk_mu_bars, width,
                 yerr=dk_se_bars, bottom=stud_mu_bars,
                 label='don\'t know', color=color_b, error_kw=error_kw)
    axes[ci].bar(xticks, other_mu_bars, width, yerr=other_se_bars,
                 bottom=stud_mu_bars + dk_mu_bars,
                 label='other-non-schematic', color=color_r, error_kw=error_kw)
    axes[ci].bar(xticks, other_sc_mu_bars, width, yerr=other_sc_se_bars,
                 bottom=stud_mu_bars + dk_mu_bars,
                 label='other-schematic', color=color_y, error_kw=error_kw)
    if ci == 0:
        axes[ci].set_title(
            'Decoded internal states, part 2\n(conditioned on correct encoding)\n%s' % condition)
    else:
        axes[ci].set_title(condition)
    axes[ci].set_ylabel('%')
    axes[ci].set_xticks(xticks)
    axes[ci].set_xticklabels(xticklabels)
    axes[ci].set_yticks([0, .5, 1])
    axes[ci].set_ylim([-.025, 1.025])
    axes[ci].set_xlim([-.5, 7.5])
    axes[ci].set_yticklabels([0, .5, 1])
# axes[-1].legend(loc=3)
# axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
axes[0].legend(bbox_to_anchor=(0, 2, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=-1, ncol=2)
axes[-1].set_xlabel('Schema strength')
sns.despine()
f.tight_layout()
img_name = 'mvpa-dr.png'
f.savefig(os.path.join('../figs', img_name), bbox_inches='tight')
#
# '''3. by condition, then by outcome, decoded response counts'''
# for condition in all_conditions:
#     f, axes = plt.subplots(4, 1, figsize=(9, 16))
#     for i, outcome in enumerate(all_outcome):
#         np.any(np.isnan(prop_dict_co_mu[condition][outcome]), axis=1)
#
#         # max_response_type
#         mrt_stud = prop_dict_co_mu[condition][outcome][:, 0]
#         mrt_dk = prop_dict_co_mu[condition][outcome][:, 1]
#         mrt_other = prop_dict_co_mu[condition][outcome][:, 2]
#         mrt_other_sc = prop_dict_co_mu[condition][outcome][:, 3]
#         mrt_stud_se = prop_dict_co_se[condition][outcome][:, 0]
#         mrt_dk_se = prop_dict_co_se[condition][outcome][:, 1]
#         mrt_other_se = prop_dict_co_se[condition][outcome][:, 2]
#         mrt_other_sc_se = prop_dict_co_se[condition][outcome][:, 3]
#
#         axes[i].bar(xticks, mrt_stud, width, yerr=mrt_stud_se,
#                     label='studied', color=color_g, error_kw=error_kw)
#         axes[i].bar(xticks, mrt_dk, width, yerr=mrt_dk_se,
#                     bottom=mrt_stud, label='don\'t know', color=color_b, error_kw=error_kw)
#         axes[i].bar(xticks, mrt_other, width, yerr=mrt_other_se,
#                     bottom=mrt_stud + mrt_dk, label='other-non-schematic', color=color_r,
#                     error_kw=error_kw)
#         axes[i].bar(xticks, mrt_other_sc, width, yerr=mrt_other_sc_se,
#                     bottom=mrt_stud + mrt_dk + mrt_other, label='other-schematic',
#                     color=color_y, error_kw=error_kw)
#
#         if i == 0:
#             axes[i].set_title(f'Condition = {condition}\noutcome = {outcome}')
#         else:
#             axes[i].set_title(f'outcome = {outcome}')
#         axes[i].set_ylabel('%')
#         axes[i].set_xticks(xticks)
#         axes[i].set_xticklabels(['%.2f' % dp for dp in def_prob_range])
#         axes[i].set_yticks([0, .5, 1])
#         axes[i].set_ylim([-.025, 1.025])
#         axes[i].set_xlim([-.5, 7.5])
#         axes[i].set_yticklabels([0, .5, 1])
#     axes[-1].legend()
#     axes[-1].set_xlabel('Schema strength')
#     sns.despine()
#     f.tight_layout()
#
#     img_name = 'mvpa-dr-counts-%s.png' % (condition)
#     f.savefig(os.path.join('../figs', img_name))
#
#
# '''4. by condition, then by mrt, outcome counts'''
# for condition in all_conditions:
#     f, axes = plt.subplots(4, 1, figsize=(9, 16))
#     for i, mrt in enumerate(max_response_type):
#         # max_response_type
#         o_correct = prop_dict_cm_mu[condition][mrt][:, 0]
#         o_dk = prop_dict_cm_mu[condition][mrt][:, 1]
#         o_mistake = prop_dict_cm_mu[condition][mrt][:, 2]
#         o_mistake_sc = prop_dict_cm_mu[condition][mrt][:, 3]
#
#         o_correct_se = prop_dict_cm_se[condition][mrt][:, 0]
#         o_dk_se = prop_dict_cm_se[condition][mrt][:, 1]
#         o_mistake_se = prop_dict_cm_se[condition][mrt][:, 2]
#         o_mistake_sc_se = prop_dict_cm_se[condition][mrt][:, 3]
#
#         axes[i].bar(xticks, o_correct, width, yerr=o_correct_se,
#                     label='correct', color=color_g, error_kw=error_kw)
#         axes[i].bar(xticks, o_dk, width, yerr=o_dk_se,
#                     bottom=o_correct, label='don\'t know', color=color_b, error_kw=error_kw)
#         axes[i].bar(xticks, o_mistake, width, yerr=o_mistake_se,
#                     bottom=o_correct + o_dk, label='mistake-non-schematic', color=color_r,
#                     error_kw=error_kw)
#         axes[i].bar(xticks, o_mistake_sc, width, yerr=o_mistake_sc_se,
#                     bottom=o_correct + o_dk + o_mistake, label='mistake-schematic',
#                     color=color_y, error_kw=error_kw)
#
#         if i == 0:
#             axes[i].set_title(f'Condition = {condition}\ndecoded = {mrt}')
#         else:
#             axes[i].set_title(f'decoded = {mrt}')
#         axes[i].set_ylabel('%')
#         axes[i].set_xticks(xticks)
#         axes[i].set_xticklabels(['%.2f' % dp for dp in def_prob_range])
#         axes[i].set_yticks([0, .5, 1])
#         axes[i].set_ylim([-.025, 1.025])
#         axes[i].set_xlim([-.5, 7.5])
#         axes[i].set_yticklabels([0, .5, 1])
#     axes[-1].legend()
#     axes[-1].set_xlabel('Schema strength')
#     sns.despine()
#     f.tight_layout()
#
#     img_name = 'mvpa-outcome-counts-%s.png' % (condition)
#     f.savefig(os.path.join('../figs', img_name))

'''sub-figure in paper'''

f, axes = plt.subplots(2, 2, figsize=(15, 9))
for j, condition in enumerate(['schema consistent', 'schema violated']):
    for i, mrt in enumerate(['studied', 'dk']):
        # max_response_type
        o_correct = prop_dict_cm_mu[condition][mrt][:, 0]
        o_dk = prop_dict_cm_mu[condition][mrt][:, 1]
        o_mistake = prop_dict_cm_mu[condition][mrt][:, 2]
        o_mistake_sc = prop_dict_cm_mu[condition][mrt][:, 3]
        o_correct_se = prop_dict_cm_se[condition][mrt][:, 0]
        o_dk_se = prop_dict_cm_se[condition][mrt][:, 1]
        o_mistake_se = prop_dict_cm_se[condition][mrt][:, 2]
        o_mistake_sc_se = prop_dict_cm_se[condition][mrt][:, 3]

        axes[i, j].bar(xticks, o_correct, width, yerr=o_correct_se,
                       label='correct', color=color_g, error_kw=error_kw)
        axes[i, j].bar(xticks, o_dk, width, yerr=o_dk_se,
                       bottom=o_correct, label='don\'t know', color=color_b, error_kw=error_kw)
        axes[i, j].bar(xticks, o_mistake, width, yerr=o_mistake_se,
                       bottom=o_correct + o_dk, label='mistake-non-schematic', color=color_r,
                       error_kw=error_kw)
        axes[i, j].bar(xticks, o_mistake_sc, width, yerr=o_mistake_sc_se,
                       bottom=o_correct + o_dk + o_mistake, label='mistake-schematic',
                       color=color_y, error_kw=error_kw)

        if i == 0:
            axes[i, j].set_title(
                f'{condition}\n(conditioned on correct encoding)\ndecoded = {mrt}')
        else:
            mrt_txt = 'don\'t know' if mrt == 'dk' else mrt
            axes[i, j].set_title(f'decoded = {mrt_txt}')
        axes[i, j].set_ylabel('%')
        axes[i, j].set_xticks(xticks)
        axes[i, j].set_xticklabels(xticklabels)
        axes[i, j].set_yticks([0, .5, 1])
        axes[i, j].set_ylim([-.025, 1.025])
        axes[i, j].set_xlim([-.5, 7.5])
        axes[i, j].set_yticklabels([0, .5, 1])
    # axes[-1, 0].legend()
    axes[0, 0].legend(bbox_to_anchor=(.6, 1.9, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=-2, ncol=2)
    axes[-1, 0].set_xlabel('Schema strength')
    axes[-1, 1].set_xlabel('Schema strength')
    sns.despine()
    f.tight_layout()

    img_name = 'mvpa-outcome-counts-bydr.png'
    f.savefig(os.path.join('../figs', img_name), bbox_inches='tight')


'''encoding performance - bar plots by condition'''
prop_enc_perf_mu = {c: np.zeros((len(def_prob_range), len(max_response_type)))
                    for c in all_conditions}
prop_enc_perf_se = {c: np.zeros((len(def_prob_range), len(max_response_type)))
                    for c in all_conditions}
for condi, cond in enumerate(all_conditions):
    for dpi, dp in enumerate(def_prob_range):
        enc_dict_dc_allsub = all_enc_dicts['%.2f' % dp][cond]
        enc_perf_ = np.array([list(enc_dict_dc_i.values())
                              for enc_dict_dc_i in enc_dict_dc_allsub])
        prop_enc_perf = enc_perf_ / np.sum(enc_perf_, axis=1, keepdims=True)
        prop_enc_perf_mu[cond][dpi, :], prop_enc_perf_se[cond][dpi, :] = compute_stats(
            prop_enc_perf)

f, axes = plt.subplots(3, 1, figsize=(8, 13))
for ci, condition in enumerate(all_conditions):
    stud_mu_bars = np.array(prop_enc_perf_mu[condition])[:, 0]
    dk_mu_bars = np.array(prop_enc_perf_mu[condition])[:, 1]
    other_mu_bars = np.array(prop_enc_perf_mu[condition])[:, 2]
    other_sc_mu_bars = np.array(prop_enc_perf_mu[condition])[:, 3]
    stud_se_bars = np.array(prop_enc_perf_se[condition])[:, 0]
    dk_se_bars = np.array(prop_enc_perf_se[condition])[:, 1]
    other_se_bars = np.array(prop_enc_perf_se[condition])[:, 2]
    other_sc_se_bars = np.array(prop_enc_perf_se[condition])[:, 3]

    axes[ci].bar(xticks, stud_mu_bars, width,
                 yerr=stud_se_bars, label='studied', color=color_g,
                 error_kw=error_kw)
    axes[ci].bar(xticks, dk_mu_bars, width,
                 yerr=dk_se_bars, bottom=stud_mu_bars,
                 label='don\'t know', color=color_b, error_kw=error_kw)
    axes[ci].bar(xticks, other_mu_bars, width, yerr=other_se_bars,
                 bottom=stud_mu_bars + dk_mu_bars,
                 label='other-non-schematic', color=color_r, error_kw=error_kw)
    axes[ci].bar(xticks, other_sc_mu_bars, width, yerr=other_sc_se_bars,
                 bottom=stud_mu_bars + dk_mu_bars + other_mu_bars,
                 label='other-schematic', color=color_y, error_kw=error_kw)
    if ci == 0:
        axes[ci].set_title(
            'Decoded internal states at encoding \n\n%s' % condition)
    else:
        axes[ci].set_title(condition)
    axes[ci].set_ylabel('%')
    axes[ci].set_xticks(xticks)
    axes[ci].set_xticklabels(xticklabels)
    axes[ci].set_yticks([0, .5, 1])
    axes[ci].set_yticklabels([0, .5, 1])
    axes[ci].set_ylim([-.025, 1.025])
    axes[ci].set_xlim([-.5, 7.5])

# axes[-1].legend(loc=3)
# axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
axes[0].legend(bbox_to_anchor=(0, 2, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=-1, ncol=2)
axes[-1].set_xlabel('Schema strength')
sns.despine()
f.tight_layout()
img_name = 'mvpa-enc-perf-bycond.png'
f.savefig(os.path.join('../figs', img_name), bbox_inches='tight')
