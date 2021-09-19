import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from utils.params import P
from utils.constants import TZ_COND_DICT
from utils.io import build_log_path, pickle_load_dict
from analysis import compute_acc, compute_dk, compute_stats, remove_none

warnings.filterwarnings("ignore")
sns.set(style='white', palette='colorblind', context='poster')
log_root = '../log/'

all_conds = TZ_COND_DICT.values()

# the name of the experiemnt
exp_name = 'vary-test-penalty'
penalty_train = 4
penalty_test = 4
comp_val = .4

n_subjs = 15
subj_ids = np.arange(n_subjs)
penalty_random = 1
def_prob = .25
n_def_tps = 0
# loading params
pad_len_load = -1
p_rm_ob_enc_load = .3
p_rm_ob_rcl_load = 0
attach_cond = 0
supervised_epoch = 600
epoch_load = 1000
learning_rate = 7e-4
n_branch = 4
n_param = 16
enc_size = 16
n_event_remember = 2
similarity_max_test = .9
similarity_min_test = 0
n_examples_test = 256


p = P(
    exp_name=exp_name, sup_epoch=supervised_epoch,
    n_param=n_param, n_branch=n_branch, pad_len=pad_len_load,
    enc_size=enc_size, n_event_remember=n_event_remember,
    def_prob=def_prob, n_def_tps=n_def_tps,
    penalty=penalty_train, penalty_random=penalty_random,
    p_rm_ob_enc=p_rm_ob_enc_load, p_rm_ob_rcl=p_rm_ob_rcl_load,
    cmpt=comp_val,
)
# create logging dirs
log_path, _ = build_log_path(
    0, p, log_root=log_root, mkdir=False, verbose=False
)
# load data
dir_all_subjs = os.path.dirname(log_path)
fname = 'p%d-%d.pkl' % (penalty_train, penalty_test)
data = pickle_load_dict(os.path.join(dir_all_subjs, fname))
# get data
acc_dict = data['acc_dict']
dk_dict = data['dk_dict']
mis_dict = data['mis_dict']
ma_list = data['lca_ma_list']
auc_list = data['auc_list']
n_subjs_ = len(remove_none(data['acc_dict']['DM']['mu']))
