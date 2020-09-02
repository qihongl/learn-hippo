import os
import torch
import numpy as np

from itertools import product
from models.LCALSTM_v1 import LCALSTM as Agent
from task import SequenceLearning
from exp_tz import run_tz
from utils.params import P
from utils.io import build_log_path, load_ckpt, pickle_save_dict, \
    get_test_data_dir, get_test_data_fname, load_env_metadata

log_root = '../log/'
# exp_name = '0220-v1-widesim-comp.8'
exp_name = '0717-dp'
def_prob_range = np.arange(.25, 1, .1)

for def_prob in def_prob_range:
    # for def_prob10 in np.arange(3, 10):
    #     # exp_name = '0425-schema.4-comp.8'
    #     exp_name = '0425-schema.%d-comp.8' % (def_prob10)
    #     def_prob = def_prob10 / 10
    # print(exp_name)

    # exp_name = '0425-schema.3-comp.8'
    # def_prob = .3
    n_def_tps = 8
    # def_prob = None
    # n_def_tps = 0

    seed = 0
    supervised_epoch = 600
    epoch_load = 1000
    learning_rate = 7e-4

    n_branch = 4
    n_param = 16
    enc_size = 16
    n_event_remember = 2

    n_hidden = 194
    n_hidden_dec = 128
    eta = .1

    penalty_random = 1
    # testing param, ortho to the training directory
    penalty_discrete = 1
    penalty_onehot = 0
    normalize_return = 1

    # loading params
    pad_len_load = -1
    p_rm_ob_enc_load = .3
    p_rm_ob_rcl_load = 0

    # testing params
    pad_len_test = 0
    p_test = 0
    p_rm_ob_enc_test = p_test
    p_rm_ob_rcl_test = p_test
    n_examples_test = 512

    similarity_max_test = .9
    similarity_min_test = 0
    # similarity_max_test = .9
    # similarity_min_test = .35

    '''loop over conditions for testing'''
    # slience_recall_times = [range(n_param), None]
    # slience_recall_times = [range(n_param)]
    slience_recall_times = [None]

    subj_ids = np.arange(10)

    penaltys_train = [4]
    # penaltys_test = np.array([0, 2, 4])
    penaltys_test = np.array([2])
    # penaltys_test = np.array(penaltys_train)
    # penaltys_train = [0, 1, 2, 4, 8]
    # penaltys_test = np.array([0, 1, 2, 4, 8])

    # all_conds = ['RM', 'DM', 'NM']
    all_conds = [None]
    scramble = False

    for slience_recall_time in slience_recall_times:
        for subj_id, penalty_train, fix_cond in product(subj_ids, penaltys_train, all_conds):
            print(
                f'\nsubj : {subj_id}, penalty : {penalty_train}, cond : {fix_cond}')
            print(f'slience_recall_time : {slience_recall_time}')

            penaltys_test_ = penaltys_test[penaltys_test <= penalty_train]
            # penaltys_test_ = np.arange(0, penalty_train + 1, 2)
            for fix_penalty in penaltys_test_:
                print(f'penalty_test : {fix_penalty}')

                p = P(
                    exp_name=exp_name, sup_epoch=supervised_epoch,
                    n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
                    def_prob=def_prob, n_def_tps=n_def_tps,
                    enc_size=enc_size, n_event_remember=n_event_remember,
                    penalty=penalty_train, penalty_random=penalty_random,
                    penalty_discrete=penalty_discrete, penalty_onehot=penalty_onehot,
                    normalize_return=normalize_return,
                    p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
                    n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
                    lr=learning_rate, eta=eta,
                )
                # create logging dirs
                log_path, log_subpath = build_log_path(
                    subj_id, p, log_root=log_root, mkdir=False, verbose=False
                )

                # init env
                env_data = load_env_metadata(log_subpath)
                def_path = env_data['def_path']
                p.env.def_path = def_path
                # p.env.def_prob = def_prob
                # def_prob_ = .25
                # p.update_enc_size(8)
                #
                task = SequenceLearning(
                    n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
                    p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
                    similarity_max=similarity_max_test, similarity_min=similarity_min_test,
                    similarity_cap_lag=p.n_event_remember,
                    # def_prob=def_prob, def_path=def_path
                )

                # load the agent back
                agent = Agent(
                    input_dim=task.x_dim, output_dim=p.a_dim,
                    rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
                    dict_len=p.net.dict_len
                )

                agent, optimizer = load_ckpt(
                    epoch_load, log_subpath['ckpts'], agent)

                # if data dir does not exsits ... skip
                if agent is None:
                    print('Agent DNE')
                    continue

                # training objective
                np.random.seed(seed)
                torch.manual_seed(seed)
                [results, metrics, XY] = run_tz(
                    agent, optimizer, task, p, n_examples_test,
                    supervised=False, learning=False, get_data=True,
                    fix_cond=fix_cond, fix_penalty=fix_penalty,
                    slience_recall_time=slience_recall_time, scramble=scramble
                )

                # save the data
                test_params = [fix_penalty, pad_len_test, slience_recall_time]
                test_data_dir, _ = get_test_data_dir(
                    log_subpath, epoch_load, test_params)
                test_data_fname = get_test_data_fname(
                    n_examples_test, fix_cond, scramble)
                test_data_dict = {'results': results,
                                  'metrics': metrics, 'XY': XY}
                fpath = os.path.join(test_data_dir, test_data_fname)
                pickle_save_dict(test_data_dict, fpath)
