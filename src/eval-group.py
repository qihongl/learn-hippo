import os
import torch
import pickle
import numpy as np

from itertools import product
from models.LCALSTM_v9 import LCALSTM as Agent
from task import SequenceLearning
from exp_tz import run_tz
from utils.params import P
from utils.io import build_log_path, load_ckpt, get_test_data_dir

log_root = '../log/'
exp_name = 'encsize_fixed'

seed = 0
n_examples_test = 512

# subj_id = 0
# penalty = 1
supervised_epoch = 300
epoch_load = 600
n_param = 16
n_branch = 4
n_event_remember = 4
enc_size = 16

n_hidden = 194
n_hidden_dec = 128
learning_rate = 1e-3
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
slience_recall_times = range(n_param)
# slience_recall_time = None

# subj_id = 0
# subj_ids = np.arange(7)
subj_ids = np.arange(1, 7)
penaltys = [4]
# penalty = 4
# fix_cond = None
# fix_cond = 'RM'
# all_conds = ['RM', 'DM', 'NM']
all_conds = [None]
for slience_recall_time in slience_recall_times:
    for subj_id, penalty, fix_cond in product(subj_ids, penaltys, all_conds):
        print(f'\nsubj_id : {subj_id}, penalty : {penalty}, cond : {fix_cond}')
        print(f'slience_recall_time : {slience_recall_time}')

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
            similarity_cap=similarity_cap_test
        )
        # create logging dirs
        log_path, log_subpath = build_log_path(subj_id, p, log_root=log_root)

        # load the agent back
        agent = Agent(
            input_dim=task.x_dim, output_dim=p.a_dim,
            rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
            dict_len=p.net.dict_len
        )
        agent, optimizer = load_ckpt(epoch_load, log_subpath['ckpts'], agent)

        # training objective
        np.random.seed(seed)
        torch.manual_seed(seed)
        [results, metrics, XY] = run_tz(
            agent, optimizer, task, p, n_examples_test,
            supervised=False, learning=False, get_data=True,
            cond=fix_cond,
            slience_recall_time=slience_recall_time
        )

        # save the data
        test_data_dir, test_data_fname = get_test_data_dir(
            log_subpath, epoch_load, pad_len_test, slience_recall_time,
            n_examples_test
        )
        test_data_dict = {'results': results, 'metrics': metrics, 'XY': XY}

        if fix_cond is not None:
            test_data_fname = fix_cond + test_data_fname
        fpath = os.path.join(test_data_dir, test_data_fname)

        with open(fpath, 'wb') as handle:
            pickle.dump(test_data_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
