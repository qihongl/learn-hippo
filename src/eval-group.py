import os
import torch
import numpy as np

from itertools import product
from models import LCALSTM as Agent
# from models import LCALSTM_after as Agent
# from models import LCALSTM_after as Agent
from task import SequenceLearning
from exp_tz import run_tz
from utils.params import P
from utils.io import build_log_path, load_ckpt, pickle_save_dict, \
    get_test_data_dir, get_test_data_fname, load_env_metadata
log_root = '../log/'

# exp_name = 'vary-test-penalty'
# exp_name = 'vary-test-penalty-after-ig.3'
# exp_name = 'vary-test-penalty-after-ig.3-enc8d4'
# exp_name = 'familiarity-signal'
# exp_name = 'vary-test-penalty-fixobs-rl'
exp_name = 'vary-test-penalty'
# def_prob = None
n_def_tps = 0

def_prob = .25
# n_def_tps = 8

seed = 0
supervised_epoch = 600
epoch_load = 1000

n_branch = 4
n_param = 16
enc_size = 16

enc_size_test = 16
dict_len_test = 2
rm_mid_targ = False

# enc_size_test = 8
# dict_len_test = 4
# rm_mid_targ = False

penalty_random = 1
# testing param, ortho to the training directory
attach_cond = 0
# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0

# testing params
pad_len_test = 0
p_test = 0
p_rm_ob_enc_test = p_test
p_rm_ob_rcl_test = p_test
n_examples_test = 256

similarity_max_test = .9
similarity_min_test = 0

permute_observations = True

'''loop over conditions for testing'''
subj_ids = np.arange(15)

penaltys_train = [4]
# penaltys_test = np.array([0, 2, 4])
penaltys_test = np.array([2])

slience_recall_times = [range(n_param), None]
# slience_recall_times = [None]
# slience_recall_times = [range(n_param)]
all_conds = ['RM', 'DM', 'NM']
# all_conds = ['DM']
# all_conds = [None]
scramble_options = [True, False]
# scramble_options = [False]

for scramble in scramble_options:
    for slience_recall_time in slience_recall_times:
        for subj_id, penalty_train, fix_cond in product(subj_ids, penaltys_train, all_conds):
            print(
                f'\nsubj : {subj_id}, penalty : {penalty_train}, cond : {fix_cond}')
            print(f'slience_recall_time : {slience_recall_time}')

            penaltys_test_ = penaltys_test[penaltys_test <= penalty_train]
            for fix_penalty in penaltys_test_:
                print(f'penalty_test : {fix_penalty}')

                p = P(
                    exp_name=exp_name, sup_epoch=supervised_epoch,
                    n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
                    def_prob=def_prob, n_def_tps=n_def_tps, enc_size=enc_size,
                    penalty=penalty_train, penalty_random=penalty_random,
                    attach_cond=attach_cond,
                    p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
                )
                # create logging dirs
                log_path, log_subpath = build_log_path(
                    subj_id, p, log_root=log_root, mkdir=False, verbose=False
                )
                # init env
                env_data = load_env_metadata(log_subpath)
                def_path = env_data['def_path']
                p.env.def_path = def_path
                p.update_enc_size(enc_size_test)

                task = SequenceLearning(
                    n_param=p.env.n_param, n_branch=p.env.n_branch, pad_len=pad_len_test,
                    p_rm_ob_enc=p_rm_ob_enc_test, p_rm_ob_rcl=p_rm_ob_rcl_test,
                    similarity_max=similarity_max_test, similarity_min=similarity_min_test,
                    similarity_cap_lag=p.n_event_remember, permute_observations=permute_observations
                )
                # load the agent back
                x_dim = task.x_dim
                if attach_cond != 0:
                    x_dim += 1
                agent = Agent(
                    input_dim=x_dim, output_dim=p.a_dim, rnn_hidden_dim=p.net.n_hidden,
                    dec_hidden_dim=p.net.n_hidden_dec, dict_len=dict_len_test
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
                    slience_recall_time=slience_recall_time, scramble=scramble,
                    rm_mid_targ=rm_mid_targ
                )

                # save the data
                test_params = [fix_penalty, pad_len_test, slience_recall_time]
                test_data_dir, _ = get_test_data_dir(
                    log_subpath, epoch_load, test_params)
                test_data_fname = get_test_data_fname(
                    n_examples_test, fix_cond, scramble)
                test_data_dict = {
                    'results': results, 'metrics': metrics, 'XY': XY
                }
                if enc_size_test != enc_size:
                    test_data_dir = os.path.join(
                        test_data_dir, f'enc_size_test-{enc_size_test}'
                    )
                    if not os.path.exists(test_data_dir):
                        os.makedirs(test_data_dir)
                fpath = os.path.join(test_data_dir, test_data_fname)
                pickle_save_dict(test_data_dict, fpath)
