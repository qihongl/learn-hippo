import os
import torch
import numpy as np

from itertools import product
from models.LCALSTM_v9 import LCALSTM as Agent
from task import SequenceLearning
from exp_tz import run_tz
from utils.params import P
from utils.io import build_log_path, load_ckpt, pickle_save_dict, \
    get_test_data_dir, get_test_data_fname

log_root = '../log/'
exp_name = 'metalearn-penalty'
# exp_name = 'prev_ar'

seed = 0
n_examples_test = 512

# subj_id = 0
# penalty = 1
supervised_epoch = 600
epoch_load = 900
n_param = 16
n_branch = 4
n_event_remember = 4
enc_size = 16

n_hidden = 194
n_hidden_dec = 128
learning_rate = 5e-4
eta = .1

# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = .3

# testing params
pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test

similarity_cap_test = .5

'''loop over conditions for testing'''
# slience_recall_times = range(n_param)
slience_recall_time = None

# subj_id = 0
# subj_ids = np.arange(7)
subj_ids = np.arange(6)
penaltys = [4]
# fix_penaltyes = [0, .5, 1, 2, 4]
fix_penaltyes = [2]
# penalty_train = 2
# fix_cond = None
# fix_cond = 'RM'
all_conds = ['RM', 'DM', 'NM']
# all_conds = [None]
# for slience_recall_time in slience_recall_times:
for subj_id, penalty_train, fix_cond in product(subj_ids, penaltys, all_conds):
    print(f'\nsubj : {subj_id}, penalty : {penalty_train}, cond : {fix_cond}')
    print(f'slience_recall_time : {slience_recall_time}')

    fix_penaltyes_ = [fp for fp in fix_penaltyes if fp <= penalty_train]
    for fix_penalty in fix_penaltyes:
        print(f'penalty_test : {fix_penalty}')

        p = P(
            exp_name=exp_name, sup_epoch=supervised_epoch,
            n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
            enc_size=enc_size, n_event_remember=n_event_remember,
            penalty=penalty_train,
            p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
            n_hidden=n_hidden, n_hidden_dec=n_hidden_dec,
            lr=learning_rate, eta=eta,
        )
        # create logging dirs
        log_path, log_subpath = build_log_path(
            subj_id, p, log_root=log_root, verbose=False
        )

        # init env
        task = SequenceLearning(
            n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
            p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
            similarity_cap=similarity_cap_test
        )

        # load the agent back
        agent = Agent(
            input_dim=task.x_dim+1, output_dim=p.a_dim,
            rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
            dict_len=p.net.dict_len
        )

        agent, optimizer = load_ckpt(epoch_load, log_subpath['ckpts'], agent)
        # if data dir does not exsits ... skip
        if agent is None:
            print('DNE')
            continue

        # training objective
        np.random.seed(seed)
        torch.manual_seed(seed)
        [results, metrics, XY] = run_tz(
            agent, optimizer, task, p, n_examples_test,
            supervised=False, learning=False, get_data=True,
            fix_cond=fix_cond, fix_penalty=fix_penalty,
            slience_recall_time=slience_recall_time
        )

        # save the data
        test_params = [fix_penalty, pad_len_test, slience_recall_time]
        test_data_dir, _ = get_test_data_dir(
            log_subpath, epoch_load, test_params)
        test_data_fname = get_test_data_fname(n_examples_test, fix_cond)
        test_data_dict = {'results': results, 'metrics': metrics, 'XY': XY}
        fpath = os.path.join(test_data_dir, test_data_fname)
        pickle_save_dict(test_data_dict, fpath)