'''parameter config class'''

from task.utils import sample_rand_path, sample_def_tps
from utils.constants import ALL_ENC_MODE
# import numpy as np


class P():
    def __init__(
        self,
        exp_name='test',
        subj_id=0,
        n_param=16,
        n_branch=4,
        pad_len=0,
        def_path=None,
        def_prob=None,
        n_def_tps=None,
        def_tps_even_odd=1,
        penalty=1,
        penalty_random=0,
        penalty_discrete=1,
        penalty_onehot=0,
        normalize_return=1,
        attach_cond=0,
        repeat_query=False,
        rm_ob_probabilistic=False,
        p_rm_ob_rcl=0,
        p_rm_ob_enc=0,
        mode_rm_ob_enc='partial',
        mode_rm_ob_rcl='all',
        n_mvs_tz=2,
        n_mvs_rnr=3,
        enc_size=None,
        enc_mode='cum',
        noisy_encoding=0,
        n_event_remember=2,
        dict_len=None,
        recall_func='LCA',
        kernel='cosine',
        n_hidden=194,
        n_hidden_dec=128,
        lr=7e-4,
        gamma=0,
        eta=.1,
        cmpt=.8,
        sup_epoch=None,
        n_epoch=None,
        n_example=None,
    ):
        # set encoding size to be maximal
        # T_part = n_param + pad_len
        if enc_size is None:
            enc_size = n_param
        assert 0 < enc_size <= n_param
        assert n_param % enc_size == 0
        self.n_event_remember = n_event_remember
        self.n_segments = n_param // enc_size
        if dict_len == None:
            dict_len = self.n_event_remember * self.n_segments

        if def_path is None:
            def_path = sample_rand_path(n_branch, n_param)
        if def_prob is None or def_prob == -1:
            def_prob = 1 / n_branch
        if n_def_tps is None:
            n_def_tps = n_param

        if def_tps_even_odd == 0:
            def_tps = sample_def_tps(n_param, n_def_tps)
        elif def_tps_even_odd == 1:
            if subj_id % 2 == 0:
                def_tps = [1, 0] * (n_param // 2)
            elif subj_id % 2 == 1:
                def_tps = [0, 1] * (n_param // 2)
            else:
                raise ValueError('subj id must be even or odd')
        else:
            raise ValueError('def_tps_even_odd must be 0 or 1')

        self.x_dim, self.y_dim, self.a_dim = _infer_data_dims(
            n_param, n_branch)
        self.dk_id = self.a_dim - 1
        # if the condition label (familiarity) is attached to the input...
        # if attach_cond != 0:
        #     self.x_dim += 1
        # init param classes
        self.env = env(
            exp_name, n_param, n_branch, pad_len,
            def_path, def_prob, def_tps,
            penalty, penalty_random, penalty_discrete, penalty_onehot,
            normalize_return, attach_cond,repeat_query,
            rm_ob_probabilistic,
            p_rm_ob_rcl, p_rm_ob_enc,
            mode_rm_ob_rcl, mode_rm_ob_enc,
            n_mvs_tz,
            n_mvs_rnr
        )
        self.net = net(
            recall_func, kernel, enc_mode, enc_size, noisy_encoding, dict_len,
            n_hidden, n_hidden_dec, lr, gamma, eta, cmpt,
            n_param, n_branch
        )
        self.misc = misc(sup_epoch, n_epoch, n_example)

        if penalty_onehot == 1:
            assert penalty_discrete == 1
        if penalty_discrete == 0:
            assert penalty_random == 1

        if self.env.penalty_onehot:
            self.extra_x_dim = len(self.env.penalty_range)
        else:
            self.extra_x_dim = 1

    def update_enc_size(self, enc_size):
        assert enc_size is not None
        assert 0 < enc_size <= self.env.n_param
        assert self.env.n_param % enc_size == 0
        self.n_segments = self.env.n_param // enc_size
        self.net.enc_size = enc_size
        self.net.dict_len = self.n_event_remember * self.n_segments

    def __repr__(self):
        repr_ = str(self.env.__repr__) + '\n' + str(self.net.__repr__)
        return repr_


class env():

    def __init__(
            self,
            exp_name,
            n_param, n_branch, pad_len,
            def_path, def_prob, def_tps,
            penalty, penalty_random, penalty_discrete, penalty_onehot,
            normalize_return, attach_cond, repeat_query,
            rm_ob_probabilistic,
            p_rm_ob_rcl, p_rm_ob_enc,
            mode_rm_ob_rcl, mode_rm_ob_enc,
            n_mvs_tz,
            n_mvs_rnr
    ):
        self.exp_name = exp_name
        self.n_param = n_param
        self.n_branch = n_branch
        self.pad_len = 'random' if pad_len == -1 else pad_len
        self.rm_ob_probabilistic = rm_ob_probabilistic
        self.p_rm_ob_rcl = p_rm_ob_rcl
        self.p_rm_ob_enc = p_rm_ob_enc
        self.mode_rm_ob_rcl = mode_rm_ob_rcl
        self.mode_rm_ob_enc = mode_rm_ob_enc
        self.def_path = def_path
        self.def_prob = def_prob
        self.def_tps = def_tps
        self.penalty = penalty
        self.penalty_random = _zero_one_to_true_false(penalty_random)
        self.penalty_discrete = _zero_one_to_true_false(penalty_discrete)
        self.penalty_onehot = _zero_one_to_true_false(penalty_onehot)
        # self.penalty_range = [i for i in range(penalty+1) if i % 2 == 0]
        self.penalty_range = [i for i in range(penalty + 1)]
        self.normalize_return = _zero_one_to_true_false(normalize_return)
        # self.attach_cond = True if attach_cond == 1 else False
        self.attach_cond = attach_cond
        self.repeat_query = repeat_query
        #
        self.chance = 1 / n_branch
        self.validate_args()

    def validate_args(self):
        assert self.penalty >= 0

    def __repr__(self):
        repr_ = f'''
        exp_name = {self.exp_name}
        n_param = {self.n_param}, n_branch = {self.n_branch},
        p_remove_observation = {self.p_rm_ob_rcl}
        penalty = {self.penalty}
        def_tps = {self.def_tps}
        def_prob = {self.def_prob}
        def_path = {self.def_path}
        '''
        return repr_


class net():
    def __init__(
        self,
        recall_func, kernel,
        enc_mode, enc_size, noisy_encoding, dict_len,
        n_hidden, n_hidden_dec, lr, gamma, eta, cmpt,
        n_param, n_branch
    ):
        self.recall_func = recall_func
        self.kernel = kernel
        self.enc_mode = enc_mode
        self.enc_size = enc_size
        self.noisy_encoding = noisy_encoding
        self.n_hidden = n_hidden
        self.n_hidden_dec = n_hidden_dec
        self.lr = lr
        self.gamma = gamma
        self.eta = eta
        self.cmpt = cmpt
        self.dict_len = dict_len
        if noisy_encoding == 1:
            self.dict_len *= 2
        # inferred params
        self.x_dim, self.y_dim, self.a_dim = _infer_data_dims(
            n_param, n_branch)
        self.dk_id = self.a_dim - 1
        self.validate_args()

    def validate_args(self):
        assert 0 <= self.gamma <= 1
        assert self.enc_mode in ALL_ENC_MODE

    def __repr__(self):
        repr_ = f'''
        recall_func = {self.recall_func}, kernel = {self.kernel}
        enc_mode = {self.enc_mode}, enc_size = {self.enc_size}
        n_hidden = {self.n_hidden}
        lr = {self.lr}
        gamma = {self.gamma}
        '''
        return repr_


class misc():

    def __init__(self, sup_epoch, n_epoch=None, n_example=None):
        self.sup_epoch = sup_epoch
        self.n_epoch = n_epoch
        self.n_example = n_example


"""helper functions"""


def get_event_ends(T_part, n_repeats):
    """get the end points for a event sequence, with lenth T, and k repeats
    - event ends need to be removed for prediction accuracy calculation, since
    there is nothing to predict there
    - event boundaries are defined by these values

    Parameters
    ----------
    T_part : int
        the length of an event sequence (one repeat)
    n_repeats : int
        number of repeats

    Returns
    -------
    1d np.array
        the end points of event seqs

    """
    return [T_part * (k + 1) - 1 for k in range(n_repeats)]


def _infer_data_dims(n_param, n_branch):
    # infer params
    x_dim = (n_param * n_branch) * 2 + n_branch
    y_dim = n_branch
    a_dim = n_branch + 1
    return x_dim, y_dim, a_dim


def _zero_one_to_true_false(int_val):
    if int_val == 0:
        bool_val = False
    elif int_val == 1:
        bool_val = True
    else:
        raise ValueError('Invalid input = {penalty_random}')
    return bool_val
