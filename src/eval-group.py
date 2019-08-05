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
# exp_name = 'penalty-fixed-discrete-simple-smalllr'
exp_name = 'penalty-fixed-discrete-simple_'
# exp_name = 'penalty-fixed-discrete-lessevent'

seed = 0

# supervised_epoch = 700
# epoch_load = 1000
# learning_rate = 6e-4
# supervised_epoch = 400
# epoch_load = 700
supervised_epoch = 300
epoch_load = 600
learning_rate = 1e-3

n_branch = 3
n_param = 16
enc_size = 16
n_event_remember = 4

n_hidden = 194
n_hidden_dec = 128
eta = .1

penalty_random = 0
# testing param, ortho to the training directory
penalty_discrete = 1
penalty_onehot = 0
normalize_return = 1

# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = .3

# testing params
pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
n_examples_test = 256

similarity_cap_test = .6

'''loop over conditions for testing'''
# slience_recall_times = range(n_param)
slience_recall_time = None

# subj_id = 0
subj_ids = np.arange(6)
# subj_ids = [0, 1]
penaltys_train = [0, 2, 4]
penaltys_test = [0, 2, 4]


# all_conds = ['RM', 'DM', 'NM']
all_conds = [None]
# for slience_recall_time in slience_recall_times:
for subj_id, penalty_train, fix_cond in product(subj_ids, penaltys_train, all_conds):
    print(f'\nsubj : {subj_id}, penalty : {penalty_train}, cond : {fix_cond}')
    print(f'slience_recall_time : {slience_recall_time}')

    # penaltys_test_ = [fp for fp in penaltys_test if fp <= penalty_train]
    penaltys_test_ = [penalty_train]
    for fix_penalty in penaltys_test_:
        print(f'penalty_test : {fix_penalty}')

        p = P(
            exp_name=exp_name, sup_epoch=supervised_epoch,
            n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
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
        task = SequenceLearning(
            n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
            p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
            similarity_cap=similarity_cap_test
        )

        # load the agent back
        agent = Agent(
            input_dim=task.x_dim+p.extra_x_dim, output_dim=p.a_dim,
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
