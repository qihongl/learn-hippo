'''parameter config class'''

from data.utils import sample_rand_path
from utils.constants import ALL_ENC_MODE

# RM, DM, NM
DISTRIBUTION_TZ_CONDITIONS = [.25, .25, .5]
# DISTRIBUTION_TZ_CONDITIONS = [1, 0, 0]


class P():
    def __init__(
        self,
        exp_name='rnr',
        n_param=11,
        n_branch=3,
        pad_len=1,
        def_path=None,
        def_prob=None,
        penalty=1,
        rm_ob_probabilistic=False,
        p_rm_ob_rcl=0,
        p_rm_ob_enc=0,
        mode_rm_ob_enc='partial',
        mode_rm_ob_rcl='all',
        n_mvs_tz=2,
        p_tz_cond=DISTRIBUTION_TZ_CONDITIONS,
        n_mvs_rnr=3,
        enc_size=None,
        enc_mode='cum',
        recall_func='LCA',
        kernel='cosine',
        n_hidden=128,
        lr=1e-3,
        gamma=0,
    ):
        # set encoding size to be maximal
        event_len = n_param + pad_len
        if enc_size is None:
            enc_size = event_len
        if def_path is None:
            def_path = sample_rand_path(n_branch, n_param)
        if def_prob is None:
            def_prob = 1/n_branch

        # infer params
        self.ohv_dim = n_param * n_branch
        state_dim = self.ohv_dim * 2
        n_action = self.ohv_dim + 1
        dk_id = n_action-1

        # init param classes
        self.env = env(
            exp_name, n_param, n_branch, pad_len,
            def_path, def_prob, penalty,
            rm_ob_probabilistic,
            p_rm_ob_rcl, p_rm_ob_enc,
            mode_rm_ob_rcl, mode_rm_ob_enc,
            n_mvs_tz, p_tz_cond,
            n_mvs_rnr
        )
        self.net = net(
            recall_func, kernel, enc_mode, enc_size,
            n_hidden, lr, gamma,
            state_dim, n_action, dk_id
        )

    def __repr__(self):
        repr_ = str(self.env.__repr__) + '\n' + str(self.net.__repr__)
        return repr_


class env():

    def __init__(
            self,
            exp_name,
            n_param, n_branch, pad_len,
            def_path, def_prob,
            penalty,
            rm_ob_probabilistic,
            p_rm_ob_rcl, p_rm_ob_enc,
            mode_rm_ob_rcl, mode_rm_ob_enc,
            n_mvs_tz, p_tz_cond,
            n_mvs_rnr
    ):
        self.exp_name = exp_name
        self.n_param = n_param
        self.n_branch = n_branch
        self.pad_len = pad_len
        self.rm_ob_probabilistic = rm_ob_probabilistic
        self.p_rm_ob_rcl = p_rm_ob_rcl
        self.p_rm_ob_enc = p_rm_ob_enc
        self.mode_rm_ob_rcl = mode_rm_ob_rcl
        self.mode_rm_ob_enc = mode_rm_ob_enc
        self.def_path = def_path
        self.def_prob = def_prob
        self.penalty = penalty
        #
        self.event_len = n_param + pad_len
        self.chance = 1 / n_branch
        #
        self.tz = tz(n_mvs_tz, self.event_len, p_tz_cond)
        self.rnr = rnr(n_mvs_rnr, self.event_len)
        self.validate_args()

    def validate_args(self):
        assert self.penalty >= 0
        assert self.pad_len >= 1

    def __repr__(self):
        repr_ = f'''
        exp_name = {self.exp_name}
        n_param = {self.n_param}, n_branch = {self.n_branch},
        pad_len = {self.pad_len}
        p_remove_observation = {self.p_rm_ob_rcl}
        def_prob = {self.def_prob}
        penalty = {self.penalty}
        def_path = {self.def_path}
        n_movies: rnr = {self.rnr.n_mvs}
        n_movies: tz  = {self.tz.n_mvs}
        '''
        return repr_


class tz():
    def __init__(self, n_mvs, event_len, p_cond):
        self.n_mvs = n_mvs
        self.event_len = event_len
        self.total_len = n_mvs * event_len
        self.event_ends = get_event_ends(event_len, n_mvs)
        self.p_cond = p_cond


class rnr():
    def __init__(self, n_mvs, event_len):
        self.n_mvs = n_mvs
        self.event_len = event_len
        self.total_len = n_mvs * event_len
        self.event_ends = get_event_ends(event_len, n_mvs)


class net():
    def __init__(
        self,
        recall_func, kernel,
        enc_mode, enc_size,
        n_hidden, lr, gamma,
        state_dim, n_action, dk_id
    ):
        self.recall_func = recall_func
        self.kernel = kernel
        self.enc_mode = enc_mode
        self.enc_size = enc_size
        self.n_hidden = n_hidden
        self.lr = lr
        self.gamma = gamma
        # inferred params
        self.state_dim = state_dim
        self.n_action = n_action
        self.dk_id = dk_id
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


"""helper functions"""


def get_event_ends(event_len, n_repeats):
    """get the end points for a event sequence, with lenth T, and k repeats
    - event ends need to be removed for prediction accuracy calculation, since
    there is nothing to predict there
    - event boundaries are defined by these values

    Parameters
    ----------
    event_len : int
        the length of an event sequence (one repeat)
    n_repeats : int
        number of repeats

    Returns
    -------
    1d np.array
        the end points of event seqs

    """
    return [event_len * (k+1)-1 for k in range(n_repeats)]
