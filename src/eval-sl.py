import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from models.LCALSTM_v9 import LCALSTM as Agent
# from models.LCALSTM_v9 import LCALSTM as Agent
# from models.LCALSTM_v9 import LCALSTM as Agent
# from models import LCALSTM as Agent
from itertools import product
from scipy.stats import pearsonr
from task import SequenceLearning
# from exp_tz import run_tz
from utils.params import P
from utils.utils import to_sqnp, to_np, to_sqpth, to_pth
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, load_ckpt, get_test_data_dir, pickle_load_dict
from analysis import compute_acc, compute_dk, compute_stats, \
    compute_trsm, compute_cell_memory_similarity, create_sim_dict, \
    compute_auc_over_time, compute_event_similarity, batch_compute_true_dk, \
    process_cache, get_trial_cond_ids, compute_n_trials_to_skip,\
    compute_cell_memory_similarity_stats, sep_by_qsource, prop_true, \
    get_qsource, trim_data

from vis import plot_pred_acc_full, plot_pred_acc_rcl, get_ylim_bonds,\
    plot_time_course_for_all_conds
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition.pca import PCA
# plt.switch_backend('agg')

sns.set(style='white', palette='colorblind', context='talk')

log_root = '../log/'
exp_name = 'encsize_fixed'
# exp_name = 'july9_v9'

# subj_ids = np.arange(7)
subj_ids = [0, 1]
penaltys = [1]

for subj_id, penalty in product(subj_ids, penaltys):

    # subj_id = 0
    # penalty = 4
    supervised_epoch = 300
    epoch_load = 600
    # n_epoch = 500
    n_param = 16
    n_branch = 4
    enc_size = 8
    n_event_remember = 4

    n_hidden = 194
    n_hidden_dec = 128
    learning_rate = 1e-3
    eta = .1

    # loading params
    p_rm_ob_enc_load = .3
    p_rm_ob_rcl_load = .3
    pad_len_load = -1
    # testing params
    p_test = 0
    p_rm_ob_enc_test = p_test
    p_rm_ob_rcl_test = p_test
    pad_len_test = 0

    slience_recall_time = None
    # slience_recall_time = 2

    n_examples_test = 512

    np.random.seed(subj_id)
    torch.manual_seed(subj_id)

    '''init'''
    p = P(
        exp_name=exp_name, sup_epoch=supervised_epoch,
        n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
        enc_size=enc_size, n_event_remember=n_event_remember,
        penalty=penalty,
        p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
        n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
        lr=learning_rate, eta=eta,
    )
    # init env
    task = SequenceLearning(
        n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
        p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
    )
    # create logging dirs
    log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)

    # # load the agent back
    # agent = Agent(
    #     input_dim=task.x_dim, output_dim=p.a_dim,
    #     rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
    #     dict_len=p.net.dict_len
    # )
    # agent, optimizer = load_ckpt(epoch_load, log_subpath['ckpts'], agent)
    # show_weight_stats(agent)
    # # training objective
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # [results, metrics, XY] = run_tz(
    #     agent, optimizer, task, p, n_examples_test,
    #     supervised=False, learning=False, get_data=True,
    #     slience_recall_time=slience_recall_time
    # )
    # log_subpath['figs']

    test_data_dir, test_data_fname = get_test_data_dir(
        log_subpath, epoch_load, pad_len_test, slience_recall_time,
        n_examples_test)
    test_data_dict = pickle_load_dict(
        os.path.join(test_data_dir, test_data_fname)
    )
    results = test_data_dict['results']
    XY = test_data_dict['XY']

    [dist_a_, Y_, log_cache_, log_cond_] = results
    [X_raw, Y_raw] = XY

    # compute ground truth / objective uncertainty (delay phase removed)
    true_dk_wm_, true_dk_em_ = batch_compute_true_dk(X_raw, task)

    '''precompute some constants'''
    # figure out max n-time-steps across for all trials
    T_part = n_param + pad_len_test
    T_total = T_part * task.n_parts
    #
    n_conds = len(TZ_COND_DICT)
    memory_types = ['targ', 'lure']
    ts_predict = np.array([t % T_part >= pad_len_test for t in range(T_total)])

    '''organize results to analyzable form'''
    # skip examples untill EM is full
    n_examples_skip = compute_n_trials_to_skip(log_cond_, p)
    n_examples = n_examples_test - n_examples_skip
    data_to_trim = [dist_a_, Y_, log_cond_,
                    log_cache_, true_dk_wm_, true_dk_em_]
    [dist_a, Y, log_cond, log_cache, true_dk_wm, true_dk_em] = trim_data(
        n_examples_skip, data_to_trim)
    # process the data
    cond_ids = get_trial_cond_ids(log_cond)
    activity_, ctrl_param_ = process_cache(log_cache, T_total, p)
    [C, H, M, CM, DA, V] = activity_
    [inpt, leak, comp] = ctrl_param_

    # onehot to int
    actions = np.argmax(dist_a, axis=-1)
    targets = np.argmax(Y, axis=-1)

    # compute performance
    corrects = targets == actions
    dks = actions == p.dk_id
    mistakes = np.logical_and(targets != actions, ~dks)

    # %%
    '''plotting params'''
    alpha = .5
    n_se = 3
    # colors
    gr_pal = sns.color_palette('colorblind')[2:4]
    # make dir to save figs
    fig_dir = os.path.join(log_subpath['figs'], f'delay-{pad_len_test}')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    '''plot behavioral performance'''

    for cn in list(TZ_COND_DICT.values()):
        Y_ = Y[cond_ids[cn], :]
        dist_a_ = dist_a[cond_ids[cn], :]
        # compute performance for this condition
        acc_mu, acc_er = compute_acc(Y_, dist_a_, return_er=True)
        dk_mu = compute_dk(dist_a_)
        # plot
        f, ax = plt.subplots(1, 1, figsize=(8, 4))
        plot_pred_acc_full(
            acc_mu, acc_er, acc_mu+dk_mu,
            [n_param], p, f, ax,
            title=f'Performance on the TZ task: {cn}',
        )
        fig_path = os.path.join(fig_dir, f'tz-acc-{cn}.png')
        f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''compare LCA params across conditions'''

    f, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    plot_time_course_for_all_conds(
        inpt, cond_ids, axes[0], axis1_start=T_part,
        title='"need" for episodic memories', ylabel='input strength'
    )
    plot_time_course_for_all_conds(
        leak, cond_ids, axes[1], axis1_start=T_part,
        title='leakiness of the memories', ylabel='leak'
    )
    plot_time_course_for_all_conds(
        comp, cond_ids, axes[2], axis1_start=T_part,
        title='competition across memories', ylabel='competition'
    )
    axes[-1].set_xlabel('Time, recall phase')
    axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    if pad_len_test > 0:
        for ax in axes:
            ax.axvline(pad_len_test, color='grey', linestyle='--')
    sns.despine()
    f.tight_layout()
    fig_path = os.path.join(fig_dir, f'tz-lca-param.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''compute cell-memory similarity / memory activation '''
    # compute similarity between cell state vs. memories
    sim_cos, sim_lca = compute_cell_memory_similarity(C, V, inpt, leak, comp)
    sim_cos_dict = create_sim_dict(sim_cos, cond_ids, n_targ=p.n_segments)
    sim_lca_dict = create_sim_dict(sim_lca, cond_ids, n_targ=p.n_segments)
    sim_cos_stats = compute_cell_memory_similarity_stats(sim_cos_dict, cond_ids)
    sim_lca_stats = compute_cell_memory_similarity_stats(sim_lca_dict, cond_ids)

    # plot target/lure activation for all conditions
    sim_stats_plt = {'LCA': sim_lca_stats, 'cosine': sim_cos_stats}
    for ker_name, sim_stats_plt_ in sim_stats_plt.items():
        # print(ker_name, sim_stats_plt_)
        f, axes = plt.subplots(3, 1, figsize=(5, 8))
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
                axes[i].set_ylabel('Memory activation')
        axes[0].legend()
        axes[-1].set_xlabel('Time, recall phase')
        # make all ylims the same
        ylim_l, ylim_r = get_ylim_bonds(axes)
        for i, ax in enumerate(axes):
            ax.set_ylim([ylim_l, ylim_r])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        if pad_len_test > 0:
            for ax in axes:
                ax.axvline(pad_len_test, color='grey', linestyle='--')
        f.tight_layout()
        sns.despine()
        fig_path = os.path.join(fig_dir, f'tz-memact-{ker_name}.png')
        f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''compute q source, and check q source % for all conditions'''

    # pick a condition
    q_source_all_conds = get_qsource(true_dk_em, true_dk_wm, cond_ids, p)
    [q_source_rm_p2, q_source_dm_p2, q_source_nm_p2] = q_source_all_conds.values()
    [eo_rm_p2, wo_rm_p2, nt_rm_p2, bt_rm_p2] = q_source_rm_p2
    [eo_dm_p2, wo_dm_p2, nt_dm_p2, bt_dm_p2] = q_source_dm_p2
    [eo_nm_p2, wo_nm_p2, nt_nm_p2, bt_nm_p2] = q_source_nm_p2

    width = .85
    f, axes = plt.subplots(3, 1, figsize=(7, 10))
    for i, (cd_name, q_source_cd_p2) in enumerate(q_source_all_conds.items()):
        # unpack data
        eo_cd_p2, wo_cd_p2, nt_cd_p2, bt_cd_p2 = q_source_cd_p2
        axes[i].bar(range(n_param), prop_true(
            eo_cd_p2), label='EM', width=width)
        axes[i].bar(range(n_param), prop_true(wo_cd_p2), label='WM', width=width,
                    bottom=prop_true(eo_cd_p2))
        axes[i].bar(range(n_param), prop_true(bt_cd_p2), label='both', width=width,
                    bottom=prop_true(eo_cd_p2)+prop_true(wo_cd_p2))
        axes[i].bar(range(n_param), prop_true(nt_cd_p2), label='neither', width=width,
                    bottom=prop_true(eo_cd_p2)+prop_true(wo_cd_p2)+prop_true(bt_cd_p2))
        axes[i].set_ylabel('Proportion (%)')
        axes[i].set_title(f'{cd_name}')
        axes[i].legend()
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[-1].set_xlabel('Time, recall phase')
    sns.despine()
    f.tight_layout()
    fig_path = os.path.join(fig_dir, f'tz-q-source.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''plot target memory activation profile, for all trials'''
    cond_name = 'DM'
    # get target memory activation
    targ_act_cond_p2 = np.mean(
        sim_lca_dict[cond_name]['targ'][:, T_part:], axis=-1)
    ylab = 'Activation'

    f, axes = plt.subplots(2, 2, figsize=(9, 7))
    # np.shape(targ_act_cond_p2)
    mu_, er_ = compute_stats(targ_act_cond_p2, n_se=3)
    axes[0, 0].plot(targ_act_cond_p2.T, alpha=.1, color=gr_pal[0])
    axes[0, 0].errorbar(x=range(T_part), y=mu_, yerr=er_, color='black')
    axes[0, 0].set_xlabel('Time, recall phase')
    axes[0, 0].set_ylabel(ylab)
    axes[0, 0].set_title(f'All trials')

    n_trials_ = 5
    trials_ = np.random.choice(range(np.shape(targ_act_cond_p2)[0]), n_trials_)
    axes[0, 1].plot(targ_act_cond_p2[trials_, :].T)

    axes[0, 1].set_xlabel('Time, recall phase')
    axes[0, 1].set_ylabel(ylab)
    axes[0, 1].set_title(f'{n_trials_} example trials')
    axes[0, 1].set_ylim(axes[0, 0].get_ylim())

    sorted_targ_act_cond_p2 = np.sort(targ_act_cond_p2, axis=1)[:, ::-1]
    mu_, er_ = compute_stats(sorted_targ_act_cond_p2, n_se=3)
    axes[1, 0].plot(sorted_targ_act_cond_p2.T, alpha=.1, color=gr_pal[0])
    axes[1, 0].errorbar(x=range(T_part), y=mu_, yerr=er_, color='black')
    axes[1, 0].set_ylabel(ylab)
    axes[1, 0].set_xlabel(f'Time (the sorting axis)')
    axes[1, 0].set_title(f'Sorted')

    recall_peak_times = np.argmax(targ_act_cond_p2, axis=1)
    sns.violinplot(recall_peak_times, color=gr_pal[0], ax=axes[1, 1])
    axes[1, 1].set_xlim(axes[0, 1].get_xlim())
    axes[1, 1].set_title(f'Max distribution')
    axes[1, 1].set_xlabel('Time, recall phase')
    axes[1, 1].set_ylabel('Density')

    if pad_len_test > 0:
        axes[0, 0].axvline(pad_len_test, color='grey', linestyle='--')
        axes[0, 1].axvline(pad_len_test, color='grey', linestyle='--')
        axes[1, 1].axvline(pad_len_test, color='grey', linestyle='--')

    f.suptitle('Activation profile of the target memory, DM',
               y=.95, fontsize=18)

    sns.despine()
    f.tight_layout(rect=[0, 0, 1, 0.9])
    # f.subplots_adjust(top=0.9)
    fig_path = os.path.join(fig_dir, f'tz-{cond_name}-targact-lca.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    # nthres = 50
    # t = 2
    # tma_min, tma_max = np.min(targ_act_cond_p2), np.max(targ_act_cond_p2)
    # thres = np.linspace(tma_min, tma_max, nthres)
    # prob = np.zeros((T_part, nthres))
    #
    # for t in range(T_part):
    #     for i, thres_i in enumerate(thres):
    #         prob[t, i] = np.mean(targ_act_cond_p2[:, t] > thres_i)
    #
    # v_pal = sns.color_palette('Blues', n_colors=T_part)
    # sns.palplot(v_pal)
    #
    # f, ax = plt.subplots(1, 1, figsize=(5, 4))
    # for t in range(T_part):
    #     ax.plot(thres, prob[t, :], color=v_pal[t])
    # ax.set_ylabel('P(activation > v | t)')
    # ax.set_xlabel('v')
    # f.tight_layout()
    # sns.despine()

    # use previous uncertainty to predict memory activation
    cond_name = 'DM'
    dk_cond_p2 = dks[cond_ids[cond_name], n_param:]
    t_pick_max = 7
    v_pal = sns.color_palette('viridis', n_colors=t_pick_max)
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for t_pick_ in range(t_pick_max):
        # compute number of don't knows produced so far
        ndks_p2_b4recall = np.sum(dk_cond_p2[:, :t_pick_], axis=1)
        nvs = np.unique(ndks_p2_b4recall)
        ma_mu = np.zeros(len(nvs),)
        ma_er = np.zeros(len(nvs),)
        for i, val in enumerate(np.unique(ndks_p2_b4recall)):
            mem_act_recall_ndk = targ_act_cond_p2[ndks_p2_b4recall == val, t_pick_]
            ma_mu[i], ma_er[i] = compute_stats(mem_act_recall_ndk, n_se=1)
        ax.errorbar(x=nvs, y=ma_mu, yerr=ma_er, color=v_pal[t_pick_])

    ax.legend(range(t_pick_max), bbox_to_anchor=(1.3, 1))
    ax.set_title('Recall ~ subjective uncertainty')
    ax.set_xlabel('# don\'t knows')
    ax.set_ylabel('average recall peak')
    sns.despine()
    f.tight_layout()
    fig_path = os.path.join(fig_dir, f'tz-{cond_name}-targact-by-ndk.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    # use objective uncertainty metric to decompose LCA params in the EM condition
    cond_name = 'DM'
    inpt_cond_p2 = inpt[cond_ids[cond_name], T_part:]
    leak_cond_p2 = leak[cond_ids[cond_name], T_part:]
    comp_cond_p2 = comp[cond_ids[cond_name], T_part:]
    q_source_cond_p2 = q_source_all_conds[cond_name]
    all_lca_param_cond_p2 = {
        'input strength': inpt_cond_p2,
        'leak': leak_cond_p2, 'competition': comp_cond_p2
    }

    f, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    for i, (param_name, param_cond_p2) in enumerate(all_lca_param_cond_p2.items()):
        # print(i, param_name, param_cond_p2)
        param_cond_p2_stats = sep_by_qsource(
            param_cond_p2[:, pad_len_test:], q_source_cond_p2, n_se=n_se)
        for key, [mu_, er_] in param_cond_p2_stats.items():
            if not np.all(np.isnan(mu_)):
                axes[i].errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
        axes[i].set_ylabel(f'{param_name}')
        axes[i].legend(fancybox=True)

    axes[-1].set_xlabel('Time, recall phase')
    axes[0].set_title(f'LCA params, {cond_name}')
    f.tight_layout()
    sns.despine()
    fig_path = os.path.join(fig_dir, f'tz-lca-param-{cond_name}.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    # use CURRENT uncertainty to predict memory activation
    targ_act_cond_p2_stats = sep_by_qsource(
        targ_act_cond_p2[:, pad_len_test:], q_source_dm_p2, n_se=n_se)

    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    for key, [mu_, er_] in targ_act_cond_p2_stats.items():
        if not np.all(np.isnan(mu_)):
            ax.errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
    ax.set_ylabel(f'{param_name}')
    ax.legend(fancybox=True)
    ax.set_title(f'Target memory activation, {cond_name}')
    ax.set_xlabel('Time, recall phase')
    ax.set_ylabel('Activation')
    ax.legend(fancybox=True)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    f.tight_layout()
    sns.despine()
    fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-qsource.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''
    predictions performance / dk / errors
    w.r.t prediction source (EM, WM)
    '''
    cond_name = 'DM'
    corrects_cond_p2 = corrects[cond_ids[cond_name], n_param:]
    mistakes_cond_p2 = mistakes[cond_ids[cond_name], n_param:]
    acc_cond_p2_stats = sep_by_qsource(
        corrects_cond_p2, q_source_dm_p2, n_se=n_se)
    dk_cond_p2_stats = sep_by_qsource(
        dk_cond_p2, q_source_dm_p2, n_se=n_se)
    mistakes_cond_p2_stats = sep_by_qsource(
        mistakes_cond_p2, q_source_dm_p2, n_se=n_se)

    stats_to_plot = {
        'correct': acc_cond_p2_stats, 'uncertain': dk_cond_p2_stats,
        'error': mistakes_cond_p2_stats,
    }

    f, axes = plt.subplots(len(stats_to_plot), 1, figsize=(7, 10))
    # loop over all statistics
    for i, (stats_name, stat) in enumerate(stats_to_plot.items()):
        # loop over all q source
        for key, [mu_, er_] in stat.items():
            # plot if sample > 0
            if not np.all(np.isnan(mu_)):
                axes[i].errorbar(x=range(n_param), y=mu_, yerr=er_, label=key)
        # for every panel/stats
        axes[i].set_ylabel(f'P({stats_name})')
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axes[i].set_ylim([-.05, 1.05])
    # for the entire panel
    axes[0].set_title(f'Performance, {cond_name}')
    axes[-1].legend(fancybox=True)
    axes[-1].set_xlabel('Time, recall phase')
    f.tight_layout()
    sns.despine()
    fig_path = os.path.join(fig_dir, f'tz-{cond_name}-stats-by-qsource.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''ma ~ correct in the EM only case'''

    tma_crt_mu, tma_crt_er = np.zeros(n_param,), np.zeros(n_param,)
    tma_incrt_mu, tma_incrt_er = np.zeros(n_param,), np.zeros(n_param,)
    for t in range(n_param):
        tma_ = targ_act_cond_p2[eo_dm_p2[:, t], t+pad_len_test]
        crt_ = corrects_cond_p2[eo_dm_p2[:, t], t]
        tma_crt_mu[t], tma_crt_er[t] = compute_stats(tma_[crt_])
        tma_incrt_mu[t], tma_incrt_er[t] = compute_stats(tma_[~crt_])

    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.errorbar(x=range(n_param), y=tma_crt_mu,
                yerr=tma_crt_er, label='correct')
    ax.errorbar(x=range(n_param), y=tma_incrt_mu,
                yerr=tma_incrt_er, label='incorrect')
    ax.set_ylim([-.05, None])
    ax.legend()
    ax.set_title(f'Target memory activation, {cond_name}')
    ax.set_ylabel('Activation')
    ax.set_xlabel('Time, recall phase')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    sns.despine()
    f.tight_layout()
    fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-cic.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''ma ~ kdk in the EM only case'''

    tma_k_mu, tma_k_er = np.zeros(n_param,), np.zeros(n_param,)
    tma_dk_mu, tma_dk_er = np.zeros(n_param,), np.zeros(n_param,)
    for t in range(n_param):
        tma_ = targ_act_cond_p2[eo_dm_p2[:, t], t+pad_len_test]
        dk_ = dk_cond_p2[eo_dm_p2[:, t], t]
        tma_k_mu[t], tma_k_er[t] = compute_stats(tma_[~dk_])
        tma_dk_mu[t], tma_dk_er[t] = compute_stats(tma_[dk_])

    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.errorbar(x=range(n_param), y=tma_k_mu, yerr=tma_k_er, label='know')
    ax.errorbar(x=range(n_param), y=tma_dk_mu,
                yerr=tma_dk_er, label='don\'t know')
    ax.set_ylim([-.05, None])
    ax.legend()
    ax.set_title(f'Target memory activation, {cond_name}')
    ax.set_ylabel('Activation')
    ax.set_xlabel('Time, recall phase')
    sns.despine()
    f.tight_layout()
    fig_path = os.path.join(fig_dir, f'tma-{cond_name}-by-kdk.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''ma ~ mistake or not, when q in neither'''
    # plt.plot(np.sum(mistakes_cond_p2, axis=0))
    #
    # # np.sum(np.logical_and(mistakes_cond_p2, eo_dm_p2), axis=0)
    # # np.sum(np.logical_and(mistakes_cond_p2, wo_dm_p2), axis=0)
    # # np.sum(np.logical_and(mistakes_cond_p2, nt_dm_p2), axis=0)
    # for t in range(n_param):
    #     print(targ_act_cond_p2[nt_dm_p2[:, t], t+pad_len_test])
    #     mistakes_cond_p2
    # # np.sum(np.logical_and(mistakes_cond_p2, bt_dm_p2), axis=0)

    '''analyze the EM-only condition'''

    f, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    plot_pred_acc_rcl(
        acc_cond_p2_stats['EM only'][0], acc_cond_p2_stats['EM only'][1],
        acc_cond_p2_stats['EM only'][0]+dk_cond_p2_stats['EM only'][0],
        p, f, ax,
        title=f'EM-based prediction performance, {cond_name}',
        baseline_on=False, legend_on=True,
    )
    if slience_recall_time is not None:
        ax.axvline(slience_recall_time, color='red',
                   linestyle='--', alpha=alpha)
    ax.set_xlabel('Time, recall phase')
    ax.set_ylabel('Accuracy')
    f.tight_layout()
    sns.despine()
    fig_path = os.path.join(fig_dir, f'tz-{cond_name}-pa-em-only.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    f, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    plot_pred_acc_rcl(
        acc_cond_p2_stats['both'][0], acc_cond_p2_stats['both'][1],
        acc_cond_p2_stats['both'][0]+dk_cond_p2_stats['both'][0],
        p, f, ax,
        title=f'WM+EM based prediction performance, {cond_name}',
        baseline_on=False, legend_on=True,
    )
    ax.set_xlabel('Time, recall phase')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([-0.05, 1.05])
    f.tight_layout()
    sns.despine()
    fig_path = os.path.join(fig_dir, f'tz-{cond_name}-pa-both.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''signal detection analysis'''

    # plot the max-score distribution for one time step
    ms_lure = np.max(sim_lca_dict['NM']['lure'], axis=-1)
    ms_targ = np.max(sim_lca_dict['DM']['targ'], axis=-1)

    leg_ = ['NM', 'DM']

    bins = 30

    sns.distplot(ms_lure[:, T_part+t])
    sns.distplot(ms_targ[:, T_part+t])

    # t s.t. maximal recall peak
    t_recall_peak = np.argmax(np.mean(targ_act_cond_p2, axis=0))
    t = t_recall_peak
    dt_ = [ms_lure[:, T_part+t], ms_targ[:, T_part+t]]

    f, ax = plt.subplots(1, 1, figsize=(6, 3))
    for j, m_type in enumerate(memory_types):
        sns.distplot(
            dt_[j],
            # hist=False,
            bins=bins,
            kde=False,
            kde_kws={"shade": True},
            ax=ax, color=gr_pal[::-1][j]
        )
    ax.legend(leg_, frameon=False,)
    ax.set_title('Max score distribution')
    ax.set_xlabel('Recall strength')
    ax.set_ylabel('Counts')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.despine()
    fig_path = os.path.join(fig_dir, f'ms-dist-t{t}.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''signal detection, max score'''
    # roc analysis
    ms_lure = ms_lure[:np.shape(ms_targ)[0], :]
    tprs, fprs, auc = compute_auc_over_time(ms_lure.T, ms_targ.T)

    b_pal = sns.color_palette('Blues', n_colors=T_part)
    f, axes = plt.subplots(2, 1, figsize=(5, 7))
    for t in np.arange(T_part, T_total):
        axes[0].plot(fprs[t], tprs[t], color=b_pal[t-T_part])
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC curves over time')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
    axes[1].plot(auc[T_part:], color='black')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('AUC over time')
    for ax in axes:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    f.tight_layout()
    sns.despine()
    fig_path = os.path.join(fig_dir, f'roc.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''compute inter-event similarity'''
    targ_raw = np.argmax(Y_raw, axis=-1)
    ambiguity = np.zeros((n_examples, n_event_remember-1))
    for i in range(n_examples):
        cur_mem_ids = np.arange(i-n_event_remember+1, i)
        for j, j_raw in enumerate(cur_mem_ids):
            ambiguity[i, j] = compute_event_similarity(targ_raw[i], targ_raw[j])

    # plot event similarity distribution
    confusion_mu = np.mean(ambiguity, axis=1)

    f, axes = plt.subplots(2, 1, figsize=(5, 6))
    sns.distplot(confusion_mu, kde=False, ax=axes[0])
    axes[0].set_ylabel('P')
    axes[0].set_xlim([0, 1])

    sns.distplot(np.ravel(ambiguity), kde=False, ax=axes[1])
    axes[1].set_xlabel('Parameter overlap')
    axes[1].set_ylabel('P')
    axes[1].set_xlim([0, 1])

    sns.despine()
    f.tight_layout()

    '''performance metrics ~ ambiguity'''
    corrects_by_cond, mistakes_by_cond, dks_by_cond = {}, {}, {}
    corrects_by_cond_mu, mistakes_by_cond_mu, dks_by_cond_mu = {}, {}, {}
    confusion_by_cond_mu = {}
    for cond_name, cond_ids_ in cond_ids.items():
        # print(cond_name, cond_ids_)
        # collect the regressor by condiiton
        confusion_by_cond_mu[cond_name] = confusion_mu[cond_ids_]
        # collect the performance metrics
        corrects_by_cond[cond_name] = corrects[cond_ids_, :]
        mistakes_by_cond[cond_name] = mistakes[cond_ids_, :]
        dks_by_cond[cond_name] = dks[cond_ids_, :]
        # compute average for the recall phase
        corrects_by_cond_mu[cond_name] = np.mean(
            corrects_by_cond[cond_name][:, T_part:], axis=1)
        mistakes_by_cond_mu[cond_name] = np.mean(
            mistakes_by_cond[cond_name][:, T_part:], axis=1)
        dks_by_cond_mu[cond_name] = np.mean(
            dks_by_cond[cond_name][:, T_part:], axis=1)

    '''show regression model w/ ambiguity as the predictor
    average the performance during the 2nd part (across time)
    '''
    # predictor: inter-event similarity
    ind_var = confusion_by_cond_mu
    dep_vars = {
        'Corrects': corrects_by_cond_mu, 'Errors': mistakes_by_cond_mu,
        'Uncertain': dks_by_cond_mu
    }
    c_pal = sns.color_palette(n_colors=3)
    f, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True)
    for col_id, cond_name in enumerate(cond_ids.keys()):
        for row_id, info_name in enumerate(dep_vars.keys()):
            sns.regplot(
                ind_var[cond_name], dep_vars[info_name][cond_name],
                # robust=True,
                scatter_kws={'alpha': .5, 'marker': '.', 's': 15},
                x_jitter=.025, y_jitter=.05,
                color=c_pal[col_id],
                ax=axes[row_id, col_id]
            )
            corr, pval = pearsonr(
                ind_var[cond_name], dep_vars[info_name][cond_name]
            )
            str_ = 'r = %.2f, p = %.2f' % (corr, pval)
            str_ = str_+'*' if pval < .05 else str_
            str_ = cond_name + '\n' + str_ if row_id == 0 else str_
            axes[row_id, col_id].set_title(str_)
            axes[row_id, 0].set_ylabel(info_name)
            axes[row_id, col_id].set_ylim([-.05, 1.05])

        axes[-1, col_id].set_xlabel('Similarity')
    sns.despine()
    f.tight_layout()
    fig_path = os.path.join(fig_dir, f'ambiguity-by-cond.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    '''show regression model w/ ambiguity as the predictor, for DM
    treat time point is indep obs (wrong), and split the data w.r.t q source
    '''
    # cond_name = 'DM'
    # confusion_cond = confusion_by_cond_mu[cond_name]
    # confusion_cond_ext = np.tile(confusion_cond, (15, 1)).T
    #
    # t = 5
    #
    # f, ax = plt.subplots(1, 1, figsize=(5, 4))
    # sns.regplot(
    #     confusion_cond_ext[eo_dm_p2[:, t], t], dk_cond_p2[eo_dm_p2[:, t], t],
    #     logistic=True,
    #     scatter_kws={'alpha': .5, 'marker': '.', 's': 15},
    #     x_jitter=.025, y_jitter=.05,
    #     ax=ax
    # )
    # ax.set_xlabel('Ambiguity')
    # ax.set_ylabel('[action]')
    # sns.despine()
    # f.tight_layout()
    # # np.mean(dk_cond_p2, axis=1)
    # # np.shape(mistakes_cond_p2)
    # np.sum(dk_cond_p2[eo_dm_p2])
    # np.sum(dk_cond_p2[~eo_dm_p2])

    '''t-RDM: raw similarity'''
    data = C
    trsm = {}
    for cond_name in cond_ids.keys():
        if np.sum(cond_ids[cond_name]) == 0:
            continue
        else:
            data_cond_ = data[cond_ids[cond_name], :, :]
            trsm[cond_name] = compute_trsm(data_cond_)

    f, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
    for i, cond_name in enumerate(TZ_COND_DICT.values()):
        sns.heatmap(
            trsm[cond_name], cmap='viridis', square=True,
            xticklabels=5, yticklabels=5,
            ax=axes[i]
        )
        axes[i].axvline(T_part, color='red', linestyle='--')
        axes[i].axhline(T_part, color='red', linestyle='--')
        axes[i].set_title(f'TR-TR correlation, {cond_name}')
        axes[i].set_ylabel('Time')
    axes[-1].set_xlabel('Time')
    f.tight_layout()
    fig_path = os.path.join(fig_dir, f'trdm-by-cond.png')
    f.savefig(fig_path, dpi=100, bbox_to_anchor='tight')

    # '''pca the deicison activity'''
    #
    # n_pcs = 5
    # data = DA
    # cond_name = 'DM'
    #
    # # fit PCA
    # pca = PCA(n_pcs)
    # # np.shape(data)
    # # np.shape(data_cond)
    # data_cond = data[cond_ids[cond_name], :, :]
    # data_cond = data_cond[:, ts_predict, :]
    # targets_cond = targets[cond_ids[cond_name]]
    # mistakes_cond = mistakes_by_cond[cond_name]
    # dks_cond = dks[cond_ids[cond_name], :]
    #
    # # Loop over timepoints
    # pca_cum_var_exp = np.zeros((np.sum(ts_predict), n_pcs))
    # for t in range(np.sum(ts_predict)):
    #     data_pca = pca.fit_transform(data_cond[:, t, :])
    #     pca_cum_var_exp[t] = np.cumsum(pca.explained_variance_ratio_)
    #
    #     f, ax = plt.subplots(1, 1, figsize=(7, 5))
    #     # plot the data
    #     for y_val in range(p.y_dim):
    #         y_sel_op = y_val == targets_cond
    #         sel_op_ = np.logical_and(
    #             ~dks[cond_ids[cond_name], t], y_sel_op[:, t])
    #         ax.scatter(
    #             data_pca[sel_op_, 0], data_pca[sel_op_, 1],
    #             marker='o', alpha=alpha,
    #         )
    #     ax.scatter(
    #         data_pca[dks[cond_ids[cond_name], t], 0],
    #         data_pca[dks[cond_ids[cond_name], t], 1],
    #         marker='o', color='grey', alpha=alpha,
    #     )
    #     ax.scatter(
    #         data_pca[mistakes_cond[:, t], 0], data_pca[mistakes_cond[:, t], 1],
    #         facecolors='none', edgecolors='red',
    #     )
    #     # add legend
    #     ax.legend(
    #         [f'choice {k}' for k in range(task.y_dim)] + ['uncertain', 'error'],
    #         fancybox=True, bbox_to_anchor=(1, .5), loc='center left'
    #     )
    #     # mark the plot
    #     ax.set_xlabel('PC 1')
    #     ax.set_ylabel('PC 2')
    #     ax.set_title(f'Pre-decision activity, time = {t}')
    #     sns.despine(offset=10)
    #     f.tight_layout()
    #
    # # plot cumulative variance explained curve
    # t = -1
    # pc_id = 1
    # f, ax = plt.subplots(1, 1, figsize=(5, 3))
    # ax.plot(pca_cum_var_exp[t])
    # ax.set_title('First %d PCs capture %d%% of variance' %
    #              (pc_id+1, pca_cum_var_exp[t, pc_id]*100))
    # ax.axvline(pc_id, color='grey', linestyle='--')
    # ax.axhline(pca_cum_var_exp[t, pc_id], color='grey', linestyle='--')
    # ax.set_ylim([None, 1.05])
    # ytickval_ = ax.get_yticks()
    # ax.set_yticklabels(['{:,.0%}'.format(x) for x in ytickval_])
    # ax.set_xticks(np.arange(n_pcs))
    # ax.set_xticklabels(np.arange(n_pcs)+1)
    # ax.set_ylabel('cum. var. exp.')
    # ax.set_xlabel('Number of components')
    # sns.despine(offset=5)
    # f.tight_layout()
    #
    # # sns.heatmap(pca_cum_var_exp, cmap='viridis')
