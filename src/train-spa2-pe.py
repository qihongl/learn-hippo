import os
import time
import torch
import pdb
# import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F
from scipy.stats import sem
from copy import deepcopy
from models import LCALSTM as Agent
# from models import LCALSTM_after as Agent
from task import SimplePairedAssociate2
from utils.utils import to_sqnp, batch_sqnp, to_np, to_pth
from utils.io import pickle_save_dict
from analysis import compute_stats
from analysis.spa import *

sns.set(style='white', palette='colorblind', context='talk')
cpals = sns.color_palette()
log_root = '../log/'
sim_name = 'cong'

'''init task and model'''
n_epochs = 600
n_cue = 16
n_assoc = 32
schema_level = .5
spa = SimplePairedAssociate2(
    n_cue=n_cue, n_assoc=n_assoc, schema_level=schema_level
)

len_study_phase = spa.n_cue * 2
T = spa.n_cue * 3
cmpt = .8
rnn_hidden_dim = 32
dec_hidden_dim = 16
lr = 3e-3
selective_encoding = False
pe_threshold = .03

subj_id = 1
n_subjs = 10
for subj_id in range(n_subjs):
    print(subj_id)
    torch.manual_seed(subj_id)
    np.random.seed(subj_id)

    agent = Agent(
        input_dim=spa.x_dim, output_dim=spa.y_dim, rnn_hidden_dim=rnn_hidden_dim,
        dec_hidden_dim=dec_hidden_dim, dict_len=n_cue, cmpt=cmpt,
        add_penalty_dim=False
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    '''training'''

    def run_model(n_epochs, train=True, recall_off=False):
        Log_loss = np.zeros((n_epochs, T))
        Log_inpt = np.zeros((n_epochs, T))
        Log_alpha = np.zeros((n_epochs, T))
        Log_acc = np.zeros((n_epochs, n_cue * 2))
        Log_cond = np.zeros((n_epochs, 2, n_cue))
        Log_ord = np.zeros((n_epochs, 2, n_cue))
        Log_t_enc = np.zeros((n_epochs, spa.n_cue))
        Log_enc_spec = np.zeros((n_epochs, 2))
        pe_c = [[] for _ in range(n_epochs)]
        pe_ic = [[] for _ in range(n_epochs)]
        miscs = [[None] * T for _ in range(n_epochs)]

        for j in range(n_epochs):
            time0 = time.time()
            spa.reset()
            X, Y, [Log_cond[j], Log_ord[j]] = spa.sample(
                return_misc=True, to_torch=True)
            # start this epoch of the task
            loss = 0
            # print(Log_cond[j][1])
            cond_std = Log_cond[j][0]

            agent.retrieval_off()
            for t in range(np.shape(X)[0]):
                hc_t = agent.get_init_states()
                # recall for the test phase
                if t >= len_study_phase:
                    agent.encoding_off()
                    if not recall_off:
                        agent.retrieval_on()
                else:
                    # study phase
                    if selective_encoding:
                        # log PE
                        if t % 2 == 1:
                            if cond_std[t // 2] == 0:
                                pe_ic[j].append(Log_loss[j, t - 1])
                            else:
                                pe_c[j].append(Log_loss[j, t - 1])

                        if t % 2 == 1 and Log_loss[j, t - 1] > pe_threshold:
                            agent.encoding_on()
                            Log_t_enc[j, t // 2] = 1
                            # print('encode', t, Log_cond[j][t//2])
                        else:
                            agent.encoding_off()
                            # print('do not', t, cond_std[t//2])
                    else:
                        if t % 2 == 1:
                            agent.encoding_on()
                            Log_t_enc[j, t // 2] = 1
                        else:
                            agent.encoding_off()

                # model forward
                pi_a_t, _, hc_t, cache = agent.forward(
                    X[t].view(1, 1, -1), hc_t)
                yhat_t = torch.squeeze(pi_a_t)
                # skip loss computation if no assoc is shown
                if t >= len_study_phase or (t < len_study_phase and t % 2 == 1):
                    loss += F.mse_loss(yhat_t, Y[t])

                # log data
                Log_loss[j, t] = F.mse_loss(yhat_t, Y[t])
                if t >= len_study_phase:
                    Log_acc[j, t -
                            n_cue] = torch.argmax(yhat_t) == np.argmax(Y[t])
                else:
                    if t % 2 == 1:
                        Log_acc[j, t //
                                2] = torch.argmax(yhat_t) == np.argmax(Y[t])

                [_, scalar_signal, miscs[j][t]] = cache
                Log_inpt[j, t], _, Log_alpha[j, t] = scalar_signal

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
                optimizer.step()
                # scheduler.step(loss)

            agent.em.reset_memory()
            time_j = time.time() - time0

            if j % 50 == 0:
                print('Epoch = %2d | Loss = %.2f | Time = %.2f' %
                      (j, loss, time_j))
        return Log_loss, Log_cond, Log_ord, Log_t_enc, Log_inpt, Log_acc, pe_c, pe_ic, miscs

    '''train the model '''

    Log_loss, Log_cond, Log_ord, Log_t_enc, Log_inpt, Log_acc, pe_c, pe_ic, miscs = run_model(
        n_epochs, train=True)

    '''plots'''

    def sep_data(data, cond):
        data_cong = deepcopy(data)
        data_incong = deepcopy(data)
        data_cong[cond == 0] = np.nan
        data_incong[cond == 1] = np.nan
        return data_cong, data_incong

    def make_lines(log_data, log_condition, f, ax):
        d_cong_tst, d_incong_tst = sep_data(
            log_data[:, n_cue:], log_condition[:, 1, :])
        d_cong_std, d_incong_std = sep_data(
            log_data[:, :n_cue], log_condition[:, 0, :])
        ax.plot(nansmooth_mean(d_cong_tst, axis=1),
                label='schematic, test')
        ax.plot(nansmooth_mean(d_incong_tst, axis=1),
                label='control, test')
        ax.plot(nansmooth_mean(d_cong_std, axis=1),
                label='schematic, study')
        ax.plot(nansmooth_mean(d_incong_std, axis=1),
                label='control, study')
        ax.set_xlabel('Epochs')
        ax.legend()
        sns.despine()
        return f, ax

    # plot the PE
    f, ax = plt.subplots(1, 1, figsize=(9, 5))
    f, ax = make_lines(Log_acc, Log_cond, f, ax)
    ax.set_ylabel('Accuracy')

    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(np.array([np.mean(pe_j) for pe_j in pe_c]), label='congruent')
    ax.plot(np.array([np.mean(pe_j)
            for pe_j in pe_ic]), label='control (enc)')
    ax.axhline(pe_threshold, linestyle='--', color='grey', label='threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PE')
    sns.despine()
    ax.legend()

    '''test phase, turn off memory'''

    def make_bars(log_data, log_condition, f, ax):
        d_cong_tst, d_incong_tst = sep_data(
            log_data[:, n_cue:], log_condition[:, 1, :])
        d_cong_std, d_incong_std = sep_data(
            log_data[:, :n_cue], log_condition[:, 0, :])

        d_list = [
            d_cong_tst, d_incong_tst,
            # d_cong_std, d_incong_std
        ]
        lgds = [
            'congruent', 'incongruent',
            # 'schematic, test', 'control, test',
            # 'schematic, study', 'control, study'
        ]
        height = [np.nanmean(d_i) for d_i in d_list]
        # print(np.shape(sem(d_cong_tst, nan_policy='omit', axis=0)))
        errorbars = np.array([
            np.nanmean(sem(d_i, nan_policy='omit', axis=0))
            for d_i in d_list
        ])
        # print(errorbars)

        ax.bar(
            x=range(len(lgds)), yerr=errorbars, height=height,
            color=cpals[:len(d_list)]
        )
        ax.set_ylim([0, 1.05])
        ax.set_xticks(range(len(lgds)))
        ax.set_xticklabels(lgds)
        sns.despine()
        return f, ax

    '''test the model '''

    n_epochs_test = 200
    Log_loss, Log_cond, Log_ord, Log_t_enc, Log_inpt, Log_acc, _, _, _ = run_model(
        n_epochs_test, train=False, recall_off=True)

    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    make_bars(Log_acc, Log_cond, f, ax)
    ax.set_ylabel('Accuracy')
    ax.set_title('Hippocampus removed')

    Log_loss, Log_cond, Log_ord, Log_t_enc, Log_inpt, Log_acc, _, _, miscs = run_model(
        n_epochs_test, train=False, recall_off=False)

    '''plot performance'''

    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    make_bars(Log_acc, Log_cond, f, ax)
    ax.set_ylabel('Accuracy')
    ax.set_title('Full model')

    '''analyis EM gate '''
    em_gate = Log_inpt[:, len_study_phase:]
    em_gate_cong, em_gate_incong = sep_data(em_gate, Log_cond[:, 1, :])

    em_gate_mu = [np.nanmean(em_gate_cong), np.nanmean(em_gate_incong)]
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar(x=range(2), height=em_gate_mu, color=cpals[:2])
    ax.set_xticks(range(2))
    ax.set_xticklabels(['cong', 'incong'])
    ax.set_ylabel('EM gate')
    f.tight_layout()
    sns.despine()

    f, ax = plt.subplots(1, 1, figsize=(3, 4))
    ax.bar(x=range(1), height=np.mean(em_gate_mu), color='grey')
    ax.set_xticks(range(1))
    ax.set_ylabel('EM gate')
    ax.set_xticks([])
    ax.set_xlabel(' ')
    f.tight_layout()
    sns.despine()

    '''memory activation analysis '''
    T_test = T - len_study_phase
    M = np.zeros((n_epochs_test, T_test, rnn_hidden_dim))
    CM = np.zeros((n_epochs_test, T_test, rnn_hidden_dim))
    DA = np.zeros((n_epochs_test, T_test, dec_hidden_dim))
    V = [[[None] for _ in range(T_test)] for _ in range(n_epochs_test)]

    # collect the data during the TEST phase
    for j in range(n_epochs_test):
        for t in np.arange(T_test):
            # shift t to collect test phase data
            tt = t + len_study_phase
            # unpack data
            [h_jt, m_jt, cm_jt, da_jt, v_jt] = miscs[j][tt]
            [h_jt, m_jt, cm_jt, da_jt] = batch_sqnp(
                [h_jt, m_jt, cm_jt, da_jt])
            # collect data
            M[j, t] = m_jt
            CM[j, t] = cm_jt
            DA[j, t] = da_jt
            V[j][t] = v_jt
    C = CM - M

    # # make sure during the test phase, #mem is the same
    # for j in range(n_epochs_test):
    #     for t in range(T_test):
    #         print(len(V[j][t]), end=' ')
    #     print()

    sim_cos, sim_lca = compute_cell_memory_similarity(C, V, em_gate, cmpt)

    # choose a similarity metric to plot
    # sim_mats = sim_cos

    def compute_targ_lure_acts(sim_mats):
        # compute the target lure act during test for cong vs. incong
        targ_act_cong = np.zeros((n_epochs_test, ))
        lure_act_cong = np.zeros((n_epochs_test, ))
        targ_act_incong = np.zeros((n_epochs_test, ))
        lure_act_incong = np.zeros((n_epochs_test, ))

        for i in range(n_epochs_test):
            targ_act_cong_i, targ_act_incong_i = [], []
            lure_act_cong_i, lure_act_incong_i = [], []

            ord_test_i = np.array(Log_ord[i][1], dtype=np.int)
            # when was each memory encoded
            mem_id_i = np.array(Log_ord[i][0], dtype=np.int)
            if selective_encoding:
                # remove memories that weren't encoded
                mem_id_i = mem_id_i[Log_t_enc[i] == 1]

            for tt, t_test in enumerate(ord_test_i):
                # use order id to identify targ vs. lure
                targ_id = t_test == mem_id_i
                lure_ids = np.logical_not(targ_id)
                # get the target activation, compute the average lure activation
                targ_act_it = sim_mats[i][tt][targ_id]
                lure_act_it = np.mean(sim_mats[i][tt][lure_ids])

                if Log_cond[i][1][tt] == 1:
                    # congruent stimuli
                    if len(targ_act_it) != 0:
                        targ_act_cong_i.append(targ_act_it)
                    lure_act_cong_i.append(lure_act_it)
                else:
                    # incongruent stimuli
                    if len(targ_act_it) != 0:
                        targ_act_incong_i.append(targ_act_it)
                    lure_act_incong_i.append(lure_act_it)

            # compute the average target/lure activation for a given trial
            targ_act_cong[i] = np.mean(targ_act_cong_i)
            targ_act_incong[i] = np.mean(targ_act_incong_i)
            lure_act_cong[i] = np.mean(lure_act_cong_i)
            lure_act_incong[i] = np.mean(lure_act_incong_i)
        return targ_act_cong, lure_act_cong, targ_act_incong, lure_act_incong

    results_ = compute_targ_lure_acts(sim_cos)
    [targ_act_cong, lure_act_cong, targ_act_incong, lure_act_incong] = results_
    # average over trials to make bar plot for this subject
    tc_cong_mu, tc_cong_se = compute_stats(targ_act_cong)
    lc_cong_mu, lc_cong_se = compute_stats(lure_act_cong)
    tc_incong_mu, tc_incong_se = compute_stats(targ_act_incong)
    lc_incong_mu, lc_incong_se = compute_stats(lure_act_incong)

    results_ = compute_targ_lure_acts(sim_lca)
    [targ_act_cong, lure_act_cong, targ_act_incong, lure_act_incong] = results_
    # average over trials to make bar plot for this subject
    ta_cong_mu, ta_cong_se = compute_stats(targ_act_cong)
    la_cong_mu, la_cong_se = compute_stats(lure_act_cong)
    ta_incong_mu, ta_incong_se = compute_stats(targ_act_incong)
    la_incong_mu, la_incong_se = compute_stats(lure_act_incong)

    mem_act_list = [ta_cong_mu, la_cong_mu, ta_incong_mu, la_incong_mu]
    mem_cos_list = [tc_cong_mu, lc_cong_mu, tc_incong_mu, lc_incong_mu]

    # compute the separation between target and lure
    tla_sep_cong = ta_cong_mu - la_cong_mu
    tla_sep_incong = ta_incong_mu - la_incong_mu

    gr_pal = sns.color_palette()[2:4]
    f, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True)
    axes[0].bar(x=range(2), height=[ta_cong_mu, la_cong_mu],
                yerr=[ta_cong_se, la_cong_se], color=gr_pal)
    axes[0].set_ylabel('Memory activation')
    axes[0].set_xticks(range(2))
    axes[0].set_xticklabels(['targ', 'lure'])
    axes[0].set_xlabel('congruent')
    axes[1].bar(x=range(2), height=[ta_incong_mu, la_incong_mu],
                yerr=[ta_incong_se, la_incong_se], color=gr_pal)
    axes[1].set_ylabel('Memory activation')
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(['targ', 'lure'])
    axes[1].set_xlabel('incongruent')
    f.tight_layout()
    sns.despine()

    ''' memory memory similarity '''
    # i = 0
    # n_cng = np.zeros(n_epochs_test, )
    # n_icg = np.zeros(n_epochs_test, )

    # # sns.heatmap(np.cov(V_i))

    # for i in range(n_epochs_test):
    #     V_i = np.array(batch_sqnp(V[i][0]))
    #     icg_mem = V_i[Log_cond[i][0] == 0]
    #     cng_mem = V_i[Log_cond[i][0] == 1]

    #     n_cng[i] = np.mean(np.linalg.norm(icg_mem, axis=1))
    #     n_icg[i] = np.mean(np.linalg.norm(cng_mem, axis=1))

    # n_cng_mu, n_cng_se = compute_stats(n_cng)
    # n_icg_mu, n_icg_se = compute_stats(n_icg)

    # f, ax = plt.subplots(1, 1, figsize=(3, 3))
    # ax.bar(x=range(2), yerr=[n_cng_se, n_icg_se], height=[
    #        n_cng_mu, n_icg_mu], color=sns.color_palette(n_colors=2))
    # ax.set_xticks(range(2))
    # ax.set_xticklabels(['congruent', 'incongruent'])
    # ax.set_ylabel('L2 norm of memories')
    # sns.despine()

    '''save data'''

    # create dir
    log_dir = os.path.join(
        log_root, 'schema-%.2f/cmpt-%.2f' % (schema_level, cmpt)
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # pack data
    data_dict = {
        'Log_loss': Log_loss, 'Log_inpt': Log_inpt,
        'Log_acc': Log_acc, 'Log_cond': Log_cond,
        'mem_act_list': mem_act_list, 'mem_cos_list': mem_cos_list,

    }
    # create data name
    enc_pol = 1 if selective_encoding else 0
    data_path = os.path.join(log_dir, f'subj-{subj_id}-enc-{enc_pol}.pkl')
    pickle_save_dict(data_dict, data_path)
