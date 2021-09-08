import pandas as pd
import os
import numpy as np
from task import SequenceLearning
from utils.params import P
from utils.utils import to_np
from utils.io import build_log_path, load_ckpt, pickle_load_dict,\
    get_test_data_dir, get_test_data_fname, load_env_metadata
from utils.constants import TZ_COND_DICT
from analysis import compute_acc, compute_dk, compute_stats, remove_none, \
    compute_cell_memory_similarity, create_sim_dict, compute_mistake, \
    batch_compute_true_dk, process_cache, get_trial_cond_ids, \
    compute_n_trials_to_skip, compute_cell_memory_similarity_stats, \
    sep_by_qsource, get_qsource, trim_data, compute_roc, get_hist_info
from analysis.task import get_oq_keys
from vis import plot_pred_acc_rcl, get_ylim_bonds
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='poster')
log_root = '../log/'
# constants
lca_pnames = {0: 'input gate', 1: 'competition'}
all_conds = list(TZ_COND_DICT.values())


# the name of the experiemnt
exp_name = 'vary-test-penalty'
penalty_train = 4
penalty_test = 4
enc_size = 16
enc_size_test = 8

n_subjs = 15
penalty_random = 1
def_prob = .25
n_def_tps = 0
cmpt = .8

# loading params
n_branch = 4
n_param = 16
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0
supervised_epoch = 600
epoch_load = 1000
similarity_max_test = .9
similarity_min_test = 0
n_examples_test = 256
pad_len_test = 0
test_data_fname = get_test_data_fname(n_examples_test, None, False)
memory_types = ['targ', 'lure']

p = P(
    exp_name=exp_name, sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
    enc_size=enc_size, def_prob=def_prob, n_def_tps=n_def_tps, cmpt=cmpt,
    penalty=penalty_train, penalty_random=penalty_random,
    p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
)
task = SequenceLearning(
    n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
    p_rm_ob_enc=0, p_rm_ob_rcl=0, similarity_cap_lag=p.n_event_remember,
    similarity_max=similarity_max_test, similarity_min=similarity_min_test,
)

# '''load data'''


def prealloc_stats():
    return {cond: {'mu': [None] * n_subjs, 'er': [None] * n_subjs}
            for cond in all_conds}


acc_dict = prealloc_stats()
mis_dict = prealloc_stats()
dk_dict = prealloc_stats()
inpt_dict = prealloc_stats()
leak_dict = prealloc_stats()
comp_dict = prealloc_stats()
ma_raw_list = [None] * n_subjs
ma_list = [None] * n_subjs
ma_cos_list = [None] * n_subjs

i_s = 0
for i_s in range(n_subjs):
    # build log path
    log_path, log_subpath = build_log_path(
        i_s, p, log_root=log_root, mkdir=False, verbose=False)
    env_data = load_env_metadata(log_subpath)
    def_path = env_data['def_path']
    def_tps = env_data['def_tps']

    test_params = [penalty_test, pad_len_test, None]
    test_data_dir, _ = get_test_data_dir(log_subpath, epoch_load, test_params)
    if enc_size_test != enc_size:
        test_data_dir += f'/enc_size_test-{enc_size_test}'
        if not os.path.exists(test_data_dir):
            print('DNE')
    fpath = os.path.join(test_data_dir, test_data_fname)
    # load data
    data_dict = pickle_load_dict(fpath)
    [dist_a_, Y_, log_cache_, log_cond_] = data_dict['results']
    [X_raw, Y_raw] = data_dict['XY']

    '''precompute some constants'''

    def_path_int = np.argmax(def_path, axis=1)
    is_def_tp = np.array(def_tps).astype(np.bool)
    # figure out max n-time-steps across for all trials
    T_part = n_param + pad_len_test
    T_total = T_part * task.n_parts
    ts_predict = np.array(
        [t % T_part >= pad_len_test for t in range(T_total)])

    '''organize results to analyzable form'''
    # skip examples untill EM is full
    n_examples_skip = compute_n_trials_to_skip(log_cond_, p)
    n_trials = n_examples_test - n_examples_skip
    trial_id = np.arange(n_trials)

    data_to_trim = [dist_a_, Y_, log_cond_, log_cache_, X_raw]
    [dist_a, Y, log_cond, log_cache, X_raw] = trim_data(
        n_examples_skip, data_to_trim)
    X_raw = np.array(X_raw)

    # process the data
    cond_ids = get_trial_cond_ids(log_cond)
    [C, H, M, CM, DA, V], [inpt] = process_cache(
        log_cache, T_total, p)
    # compute ground truth / objective uncertainty, delay phase removed
    true_dk_wm, true_dk_em = batch_compute_true_dk(X_raw, task)
    q_source = get_qsource(true_dk_em, true_dk_wm, cond_ids, p)

    # load lca params
    comp = np.full(np.shape(inpt), cmpt)
    leak = np.full(np.shape(inpt), 0)

    # compute performance
    actions = np.argmax(dist_a, axis=-1)
    targets = np.argmax(Y, axis=-1)
    corrects = targets == actions
    dks = actions == p.dk_id
    mistakes = np.logical_and(targets != actions, ~dks)

    # split data wrt p1 and p2
    CM_p1, CM_p2 = CM[:, :T_part, :], CM[:, T_part:, :]
    DA_p1, DA_p2 = DA[:, :T_part, :], DA[:, T_part:, :]
    X_raw_p1, X_raw_p2 = X_raw[:, :T_part, :], X_raw[:, T_part:, :]
    corrects_p2 = corrects[:, T_part:]
    mistakes_p1 = mistakes[:, :T_part]
    mistakes_p2 = mistakes[:, T_part:]
    dks_p1, dks_p2 = dks[:, :T_part], dks[:, T_part:]
    inpt_p2 = inpt[:, T_part:]
    targets_p1 = targets[:, :T_part]
    targets_p2 = targets[:, T_part:]
    actions_p1 = actions[:, :T_part]
    actions_p2 = actions[:, T_part:]

    # pre-extract p2 data for the DM condition
    corrects_dmp2 = corrects_p2[cond_ids['DM']]
    mistakes_dmp2 = mistakes_p2[cond_ids['DM']]
    mistakes_dmp1 = mistakes_p1[cond_ids['DM']]
    dks_dmp2 = dks_p2[cond_ids['DM']]
    CM_dmp2 = CM_p2[cond_ids['DM']]
    DA_dmp2 = DA_p2[cond_ids['DM']]

    inpt_dmp2 = inpt_p2[cond_ids['DM']]
    targets_dmp1 = targets_p1[cond_ids['DM'], :]
    targets_dmp2 = targets_p2[cond_ids['DM'], :]
    actions_dmp1 = actions_p1[cond_ids['DM']]
    actions_dmp2 = actions_p2[cond_ids['DM']]

    # get observation key and values for p1 p2
    o_keys = np.zeros((n_trials, T_total))
    o_vals = np.zeros((n_trials, T_total))
    for i in trial_id:
        o_keys[i], _, o_vals[i] = get_oq_keys(X_raw[i], task)
    o_keys_p1, o_keys_p2 = o_keys[:, :T_part], o_keys[:, T_part:]
    o_vals_p1, o_vals_p2 = o_vals[:, :T_part], o_vals[:, T_part:]
    o_keys_dmp1 = o_keys_p1[cond_ids['DM']]
    o_keys_dmp2 = o_keys_p2[cond_ids['DM']]
    o_vals_dmp1 = o_vals_p1[cond_ids['DM']]
    o_vals_dmp2 = o_vals_p2[cond_ids['DM']]

    # compute performance stats
    for i, cn in enumerate(all_conds):
        Y_ = Y[cond_ids[cn], :]
        dist_a_ = dist_a[cond_ids[cn], :]
        # compute performance for this condition
        acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
        dk_mu = compute_dk(dist_a_)
        mis_mu, mis_er = compute_mistake(
            Y_, dist_a_, return_er=True)

        # cache data for all cond-subj
        acc_dict[cn]['mu'][i_s] = acc_mu
        acc_dict[cn]['er'][i_s] = acc_er
        mis_dict[cn]['mu'][i_s] = mis_mu
        mis_dict[cn]['er'][i_s] = mis_er
        dk_dict[cn]['mu'][i_s] = dk_mu

    '''plot behavioral performance'''
    f, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, cn in enumerate(['RM', 'DM', 'NM']):
        Y_ = Y[cond_ids[cn], :]
        dist_a_ = dist_a[cond_ids[cn], :]
        # compute performance for this condition
        acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
        dk_mu = compute_dk(dist_a_)

        if i == 0:
            add_legend = True
            show_ylabel = True
            legend_loc = (.33, .7)
        else:
            add_legend = False
            show_ylabel = False

        # plot
        plot_pred_acc_rcl(
            acc_mu[T_part:], acc_er[T_part:],
            acc_mu[T_part:] + dk_mu[T_part:],
            p, f, axes[i],
            title=f'{cn}',
            add_legend=add_legend, legend_loc=legend_loc,
            show_ylabel=show_ylabel
        )
        # axes[i].set_ylabel()
        axes[i].set_ylim([-.05, 1.05])
    # fig_path = os.path.join(fig_dir, f'tz-acc-horizontal.png')
    # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''compute cell-memory similarity / memory activation '''
    lca_param_names = ['input gate', 'competition']
    lca_param_dicts = [inpt_dict, comp_dict]
    lca_param_records = [inpt, comp]
    for i, cn in enumerate(all_conds):
        for p_dict, p_record in zip(lca_param_dicts, lca_param_records):
            p_dict[cn]['mu'][i_s], p_dict[cn]['er'][i_s] = compute_stats(
                p_record[cond_ids[cn]])

    # compute similarity between cell state vs. memories
    sim_cos, sim_lca = compute_cell_memory_similarity(
        C, V, inpt, leak, comp)
    sim_cos_dict = create_sim_dict(
        sim_cos, cond_ids, n_targ=p.n_segments)
    sim_lca_dict = create_sim_dict(
        sim_lca, cond_ids, n_targ=p.n_segments)
    sim_cos_stats = compute_cell_memory_similarity_stats(
        sim_cos_dict, cond_ids)
    sim_lca_stats = compute_cell_memory_similarity_stats(
        sim_lca_dict, cond_ids)
    # save for all subjects
    ma_list[i_s] = sim_lca_stats
    ma_raw_list[i_s] = sim_lca_dict
    ma_cos_list[i_s] = sim_cos_stats
    sim_cos_dict.keys()
    f, axes = plt.subplots(4, 1, figsize=(5, 8))
    for i in range(4):
        axes[i].plot(sim_lca[0, :, i])
    f.tight_layout()

    avg_ma = {cond: {m_type: None for m_type in memory_types}
              for cond in all_conds}
    for cond in all_conds:
        for m_type in memory_types:
            if sim_lca_dict[cond][m_type] is not None:
                avg_ma[cond][m_type] = np.mean(
                    sim_lca_dict[cond][m_type], axis=-1)

    '''plot target/lure activation for all conditions - horizontal'''
    gr_pal = sns.color_palette('colorblind')[2:4]
    ylim_bonds = {'LCA': None, 'cosine': None}
    ker_name, sim_stats_plt_ = 'LCA', sim_lca_stats
    tsf = (T_part + pad_len_test) / T_part
    f, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, c_name in enumerate(cond_ids.keys()):
        for m_type in memory_types:
            if m_type == 'targ' and c_name == 'NM':
                continue
            color_ = gr_pal[0] if m_type == 'targ' else gr_pal[1]
            axes[i].errorbar(
                x=range(T_part),
                y=sim_stats_plt_[c_name][m_type]['mu'][T_part:],
                yerr=sim_stats_plt_[c_name][m_type]['er'][T_part:],
                label=f'{m_type}', color=color_
            )
            axes[i].set_title(c_name)
            axes[i].set_xlabel('Time')
    axes[0].set_ylabel('Memory activation')
    axes[0].legend()

    # make all ylims the same
    ylim_bonds[ker_name] = get_ylim_bonds(axes)
    ylim_bonds[ker_name] = (
        np.max((ylim_bonds[ker_name][0], -.05)
               ), np.round((ylim_bonds[ker_name][1] + .1), decimals=1)
    )
    for i, ax in enumerate(axes):
        ax.set_ylim(ylim_bonds[ker_name])
        ax.set_xticks([0, p.env.n_param - 1])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if pad_len_test > 0:
        for ax in axes:
            ax.axvline(pad_len_test, color='grey', linestyle='--')
    f.tight_layout()
    sns.despine()
    # fig_path = os.path.join(
    #     fig_dir, f'tz-memact-{ker_name}-hori.png')
    # f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')
