import os
import torch
import numpy as np
# from itertools import product
from models import LCALSTM as Agent
from task import SequenceLearning
from exp_aba import run_aba
from utils.params import P
from utils.io import build_log_path, load_ckpt, pickle_save_dict, \
    get_test_data_fname

# log_root = '../log/'
log_root = '/tigress/qlu/logs/learn-hippocampus/log'

# exp_name = 'penalty-random-discrete'
exp_name = '0916-widesim-prandom'

seed = 0
supervised_epoch = 600
epoch_load = 1600
learning_rate = 7e-4
n_event_remember_train = 2

n_branch = 4
n_param = 16
enc_size = 16
def_prob = None

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

'''loop over conditions for testing'''

subj_ids = np.arange(16)
penalty_train = 4
fix_penalty = 2
fix_cond = 'DM'

n_examples_test = 1024
# similarity_cap_test = .3
similarity_max_test = .9
similarity_min_test = 0

n_event_remember_test = 2
p_rm_ob = 0.4
n_parts = 3
pad_len = 0
scramble = False
slience_recall_time = None

for subj_id in subj_ids:
    print(f'subj_id = {subj_id}')
    p = P(
        exp_name=exp_name, sup_epoch=supervised_epoch,
        n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
        def_prob=def_prob,
        enc_size=enc_size, n_event_remember=n_event_remember_train,
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
        n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len,
        p_rm_ob_enc=p_rm_ob, p_rm_ob_rcl=p_rm_ob,
        similarity_cap_lag=p.n_event_remember,
        similarity_max=similarity_max_test, similarity_min=similarity_min_test,
        n_parts=n_parts
    )

    # load the agent back
    agent = Agent(
        input_dim=task.x_dim, output_dim=p.a_dim,
        rnn_hidden_dim=p.net.n_hidden, dec_hidden_dim=p.net.n_hidden_dec,
        dict_len=n_event_remember_test
    )
    log_output_path = os.path.join(
        log_subpath['ckpts'], f'n_event_remember-{n_event_remember_test}',
        f'p_rm_ob-{p_rm_ob}', f'similarity_cap-{similarity_min_test}_{similarity_max_test}')
    ckpt_path = os.path.join(log_subpath['ckpts'], 'aba')
    agent, optimizer = load_ckpt(epoch_load, ckpt_path, agent)
    # if data dir does not exsits ... skip
    if agent is None:
        print('DNE')
        continue

    # training objective
    np.random.seed(seed)
    torch.manual_seed(seed)
    [results, metrics, XY] = run_aba(
        agent, optimizer, task, p, n_examples_test,
        supervised=False, fix_cond=fix_cond, learning=False, get_data=True,
    )

    # save the data
    log_data_path = os.path.join(
        log_subpath['data'], f'n_event_remember-{n_event_remember_test}',
        f'p_rm_ob-{p_rm_ob}', f'similarity_cap-{similarity_min_test}_{similarity_max_test}')
    if not os.path.exists(log_data_path):
        os.makedirs(log_data_path)
    test_data_fname = get_test_data_fname(
        n_examples_test, fix_cond, scramble)
    test_data_dict = {'results': results, 'metrics': metrics, 'XY': XY}
    fpath = os.path.join(log_data_path, test_data_fname)
    pickle_save_dict(test_data_dict, fpath)
