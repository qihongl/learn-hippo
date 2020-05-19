import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.io import pickle_load_dict
from analysis import compute_stats
sns.set(style='white', palette='colorblind', context='poster')


def remove_all_zero_sub(d):
    n_subjs = np.shape(d)[0]
    all_zero_sub = []
    for i in range(n_subjs):
        if np.all(np.array(d)[i] == 0):
            all_zero_sub.append(i)
    d = np.delete(d, all_zero_sub, axis=0)
    # print(all_zero_sub)
    return d, all_zero_sub


schema_levels = np.arange(.3, 1, .1)
n_schema_levels = len(schema_levels)
data = [None] * n_schema_levels

corrects_wwoproto_cic_mu = np.zeros((n_schema_levels, 2, 2))
corrects_wwoproto_cic_se = np.zeros((n_schema_levels, 2, 2))
inpt_wwoproto_cic_g_mu = np.zeros((n_schema_levels, 2, 2))
inpt_wwoproto_cic_g_se = np.zeros((n_schema_levels, 2, 2))
dk_wwoproto_cic_g_mu = np.zeros((n_schema_levels, 2, 2))
dk_wwoproto_cic_g_se = np.zeros((n_schema_levels, 2, 2))

nscm_mu = np.zeros(n_schema_levels)
nscm_se = np.zeros(n_schema_levels)
nsicm_mu = np.zeros(n_schema_levels)
nsicm_se = np.zeros(n_schema_levels)
ndk_mu = np.zeros(n_schema_levels)
ndk_se = np.zeros(n_schema_levels)
ncorrect_mu = np.zeros(n_schema_levels)
ncorrect_se = np.zeros(n_schema_levels)
scmr_mu = np.zeros(n_schema_levels)
scmr_se = np.zeros(n_schema_levels)

schema_level = .4
for i, schema_level in enumerate(schema_levels):
    # print(schema_level)
    data_path = 'temp/schema-%.2f' % schema_level
    data[i] = pickle_load_dict(data_path)
    corrects_wwoproto_cic_g = data[i]['corrects']
    inpt_wwoproto_cic_g = data[i]['inpt']
    dk_wwoproto_cic_g = data[i]['dks']
    n_sc_mistakes = data[i]['n_sc_mistakes']
    n_sic_mistakes = data[i]['n_sic_mistakes']
    n_dks = data[i]['n_dks']
    n_corrects = data[i]['n_corrects']

    n_subjs = np.shape(corrects_wwoproto_cic_g)[0]

    corrects_wwoproto_cic_g, all_zero_sub = remove_all_zero_sub(
        corrects_wwoproto_cic_g)
    inpt_wwoproto_cic_g, all_zero_sub = remove_all_zero_sub(
        inpt_wwoproto_cic_g)
    dk_wwoproto_cic_g, all_zero_sub = remove_all_zero_sub(
        dk_wwoproto_cic_g)

    n_sc_mistakes_rm0 = np.delete(n_sc_mistakes, all_zero_sub)
    n_sic_mistakes_rm0 = np.delete(n_sic_mistakes, all_zero_sub)
    n_dks_rm0 = np.delete(n_dks, all_zero_sub)
    n_corrects_rm0 = np.delete(n_corrects, all_zero_sub)

    corrects_wwoproto_cic_mu[i] = np.mean(corrects_wwoproto_cic_g, axis=0)
    corrects_wwoproto_cic_se[i] = np.std(
        corrects_wwoproto_cic_g, axis=0) / np.sqrt(n_subjs)
    inpt_wwoproto_cic_g_mu[i] = np.mean(inpt_wwoproto_cic_g, axis=0)
    inpt_wwoproto_cic_g_se[i] = np.std(
        inpt_wwoproto_cic_g, axis=0) / np.sqrt(n_subjs)
    dk_wwoproto_cic_g_mu[i] = np.mean(dk_wwoproto_cic_g, axis=0)
    dk_wwoproto_cic_g_se[i] = np.std(
        dk_wwoproto_cic_g, axis=0) / np.sqrt(n_subjs)

    nscm_mu[i], nscm_se[i] = compute_stats(n_sc_mistakes_rm0)
    nsicm_mu[i], nsicm_se[i] = compute_stats(n_sic_mistakes_rm0)
    scmr = n_sc_mistakes_rm0 / (n_sc_mistakes_rm0 + n_sic_mistakes_rm0)
    scmr = scmr[~np.isnan(scmr)]
    scmr_mu[i], scmr_se[i] = compute_stats(scmr)

    ndk_mu[i], ndk_se[i] = compute_stats(n_dks_rm0)
    ncorrect_mu[i], ncorrect_se[i] = compute_stats(n_corrects_rm0)
    print(schema_level, all_zero_sub)


schema_levels_txt = ['%.1f' % schema_level for schema_level in schema_levels]

# correct rate
colors2 = sns.color_palette('colorblind')[:2]
x_ticks = np.arange(n_schema_levels)
n_se = 1

f, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.errorbar(
    x=x_ticks, y=corrects_wwoproto_cic_mu[:, 0, 0],
    yerr=corrects_wwoproto_cic_se[:, 0, 0] * n_se,
    color=colors2[0]
)
ax.errorbar(
    x=x_ticks, y=corrects_wwoproto_cic_mu[:, 0, 1],
    yerr=corrects_wwoproto_cic_se[:, 0, 1] * n_se,
    color=colors2[1]
)

ax.errorbar(
    x=x_ticks, y=np.mean(corrects_wwoproto_cic_mu[:, 1, :], axis=1),
    yerr=np.mean(corrects_wwoproto_cic_se[:, 1, :], axis=1) * n_se,
    color='k', linestyle='--',
)
# ax.axhline(.25, linestyle='--', color='black', alpha=.5)

ax.legend(['prototypical event happened',
           'prototypical event violated', 'no prototypical event'])

ax.set_ylabel('% correct')
ax.set_xlabel('Regularity strength')
ax.set_xticks(x_ticks)
ax.set_xticklabels(schema_levels_txt)
f.tight_layout()
sns.despine()

fname = f'../figs/schema-regularity-correctr.png'
f.savefig(fname, dpi=120, bbox_to_anchor='tight')

# # input gate
# colors2 = sns.color_palette('colorblind')[:2]
# x_ticks = np.arange(n_schema_levels)
# n_se = 1
#
# f, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.errorbar(
#     x=x_ticks, y=inpt_wwoproto_cic_g_mu[:, 0, 0],
#     yerr=inpt_wwoproto_cic_g_se[:, 0, 0] * n_se,
#     color=colors2[0]
# )
# ax.errorbar(
#     x=x_ticks, y=inpt_wwoproto_cic_g_mu[:, 0, 1],
#     yerr=inpt_wwoproto_cic_g_se[:, 0, 1] * n_se,
#     color=colors2[1]
# )
#
# ax.errorbar(
#     x=x_ticks, y=np.mean(inpt_wwoproto_cic_g_mu[:, 1, :], axis=1),
#     yerr=np.mean(inpt_wwoproto_cic_g_se[:, 1, :], axis=1) * n_se,
#     color='k', linestyle='--',
# )
# # ax.axhline(.25, linestyle='--', color='black', alpha=.5)
#
# ax.legend(['prototypical event happened',
#            'prototypical event violated', 'no prototypical event'])
#
# ax.set_ylabel('Input gate')
# ax.set_xlabel('Regularity strength')
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(schema_levels_txt)
# f.tight_layout()
# sns.despine()
# fname = f'../figs/schema-regularity-inpt.png'
# f.savefig(fname, dpi=120, bbox_to_anchor='tight')


# dk
# colors2 = sns.color_palette('colorblind')[:2]
# x_ticks = np.arange(n_schema_levels)
n_se = 1

f, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.errorbar(
    x=x_ticks, y=np.mean(dk_wwoproto_cic_g_mu[:, 0, :], axis=1),
    yerr=np.mean(dk_wwoproto_cic_g_se[:, 0, :], axis=1) * n_se,
    color='k'
)

ax.errorbar(
    x=x_ticks, y=np.mean(dk_wwoproto_cic_g_mu[:, 1, :], axis=1),
    yerr=np.mean(dk_wwoproto_cic_g_se[:, 1, :], axis=1) * n_se,
    color='k', linestyle='--',
)
ax.legend(['has a prototypical event', 'no prototypical event'])
ax.set_ylabel('% don\'t know responses')
ax.set_xlabel('Regularity strength')
ax.set_xticks(x_ticks)
ax.set_xticklabels(schema_levels_txt)
f.tight_layout()
sns.despine()
fname = f'../figs/schema-regularity-dk.png'
f.savefig(fname, dpi=120, bbox_to_anchor='tight')

f, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.errorbar(x=x_ticks, y=scmr_mu, yerr=scmr_se, color='k')
ax.set_ylabel('% mistakes schema-consistent')
ax.set_ylim([0, 1])
ax.set_xlabel('Regularity strength')
ax.set_xticks(x_ticks)
ax.set_xticklabels(schema_levels_txt)
f.tight_layout()
sns.despine()
fname = f'../figs/percent-schema-consistent-mistakes.png'
f.savefig(fname, dpi=120, bbox_to_anchor='tight')

#
# f, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.errorbar(x=x_ticks, y=ndk_mu, yerr=ndk_se, color=colors2[0])
# ax.set_ylabel('% mistakes schema-consistent')
# # ax.set_ylim([0, 1])
# # ax.set_xlabel('Regularity strength')
# # ax.set_xticks(x_ticks)
# # ax.set_xticklabels(schema_levels_txt)
# # f.tight_layout()
# # sns.despine()
# # fname = f'../figs/percent-schema-consistent-mistakes.png'
# # f.savefig(fname, dpi=120, bbox_to_anchor='tight')
