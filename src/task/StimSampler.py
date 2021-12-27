import numpy as np
from task.Schema import Schema
from task.utils import get_botvinick_query

# import pdb


class StimSampler():
    '''
    a sampler of sequences
    '''

    def __init__(
            self,
            n_param,
            n_branch,
            pad_len=0,
            max_pad_len=None,
            def_path=None,
            def_prob=None,
            def_tps=None,
            key_rep_type='time',
            rm_kv=False,
            context_onehot=True,
            context_dim=1,
            context_drift=False,
            n_rm_fixed=False,
            sampling_mode='enumerative',
            repeat_query=False,
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        self.pad_len = pad_len
        if max_pad_len is None:
            self.max_pad_len = np.max([n_param // 3 - 1, 0])
        #
        self.def_path = def_path
        self.def_prob = def_prob
        self.def_tps = def_tps
        #
        self.context_onehot = context_onehot
        self.context_dim = context_dim
        self.context_drift = context_drift
        #
        self.key_rep_type = key_rep_type
        self.sampling_mode = sampling_mode
        #
        self.rm_kv = rm_kv
        self.n_rm_fixed = n_rm_fixed
        #
        self.repeat_query=repeat_query
        #
        self.reset_schema()

    def reset_schema(self):
        """re-initialize the schema
        """
        self.schema = Schema(
            n_param=self.n_param,
            n_branch=self.n_branch,
            def_path=self.def_path,
            def_prob=self.def_prob,
            def_tps=self.def_tps,
            context_onehot=self.context_onehot,
            context_dim=self.context_dim,
            context_drift=self.context_drift,
            key_rep_type=self.key_rep_type,
            sampling_mode=self.sampling_mode,
        )
        self.k_dim = self.schema.k_dim
        self.v_dim = self.schema.v_dim
        self.c_dim = self.schema.c_dim

    def _sample(self):
        """sample an event sequence, one-hot vector representation

        Returns
        -------
        2d np array, 2d np array; T x (T x B), T x B
            sequence of keys / parameter values over time

        """
        # sample keys and parameter values, integer representation
        keys, vals = self.schema.sample()
        # translate to vector representation
        keys_vec = np.vstack([self.schema.key_rep[k_t, :] for k_t in keys])
        vals_vec = np.vstack([self.schema.val_rep[v_t, :] for v_t in vals])
        # ctxs_vec = np.vstack([self.schema.ctx_rep[v_t, :] for v_t in vals])
        ctxs_vec = np.vstack([self.schema.ctx_rep])
        misc = [keys, vals]
        return keys_vec, vals_vec, ctxs_vec, misc

    def sample(
            self,
            n_parts=2, p_rm_ob_enc=0, p_rm_ob_rcl=0,
            permute_observations=True, permute_queries=False,
    ):
        """sample a multi-part "movie", with repetition structure

        Parameters
        ----------
        n_parts : int
            the number of parts in this event sequence
        format: string
            the output data format
            - 'okv-qkv': human-readble form
            - 'xy': nn-readable form

        Returns
        -------
        3d np array, 3d np array; nP x T x (T x B), nP x T x B
            sequence of keys / parameter values over time
            different parts are consistent

        """
        # sample the state-param associtations
        keys_vec_, vals_vec_, ctxs_vec_, misc = self._sample()
        # sample for the observation phase
        o_keys_vec, o_vals_vec = self._sample_permutations_sup(
            keys_vec_, vals_vec_, n_parts, permute_observations)
        q_keys_vec, q_vals_vec = self._sample_permutations_sup(
            keys_vec_, vals_vec_, n_parts, permute_queries)
        # corrupt input during encoding
        o_keys_vec, o_vals_vec = self._corrupt_observations(
            o_keys_vec, o_vals_vec, p_rm_ob_enc, p_rm_ob_rcl)
        # context are assumed to repeat across the two phases
        o_ctxs_vec = q_ctxs_vec = ctxs_vec_
        # whether to repeat query
        if self.repeat_query:
            q_keys_vec = [get_botvinick_query(self.n_param) for _ in range(n_parts)]
        # pack sample
        o_sample_ = [o_keys_vec, o_vals_vec, o_ctxs_vec]
        q_sample_ = [q_keys_vec, q_vals_vec, q_ctxs_vec]
        # padding, if there is a delay
        [o_sample_, q_sample_] = self._delay_pred_demand(o_sample_, q_sample_)
        # pack sample
        sample_ = [o_sample_, q_sample_]
        return sample_, misc

    def _sample_permutations_sup(
        self, keys_vec_raw, vals_vec_raw, n_parts, permute
    ):
        if permute:
            s_keys_vec, s_vals_vec = self._sample_permutations(
                keys_vec_raw, vals_vec_raw, n_parts)
        else:
            s_keys_vec = np.stack([keys_vec_raw for _ in range(n_parts)])
            s_vals_vec = np.stack([vals_vec_raw for _ in range(n_parts)])
        return s_keys_vec, s_vals_vec

    def _sample_permutations(self, keys_vec_raw, vals_vec_raw, n_perms):
        """given some raw key-val pairs, generate temporal permutation sets
        """
        T = self.n_param
        keys_vec = np.zeros((n_perms, T, self.k_dim))
        vals_vec = np.zeros((n_perms, T, self.v_dim))
        # ctxs_vec = np.zeros((n_perms, T, self.c_dim))
        for ip in range(n_perms):
            # unique permutation for each movie part
            perm_op = np.random.permutation(T)
            keys_vec[ip] = keys_vec_raw[perm_op, :]
            vals_vec[ip] = vals_vec_raw[perm_op, :]
            # ctxs_vec[ip] = ctxs_vec_raw[perm_op, :]
        return keys_vec, vals_vec

    def _corrupt_observations(
        self,
        o_keys_vec, o_vals_vec,
        p_rm_ob_enc, p_rm_ob_rcl,
    ):
        """corrupt observations
        currently I only implemented zero-ing out random rows, but this function
        can be more general than this

        Parameters
        ----------
        o_keys_vec : 3d np array, nP x T x (T x B)
            keys, or states
        o_vals_vec : 3d np array, nP x T x B
            values, or actions
        p_rm_ob_enc : float
            p(zero-ing out observation at time t) during encoding
        p_rm_ob_rcl : float
            p(zero-ing out observation at time t) during recall

        Returns
        -------
        3d np array, 3d np array; nP x T x (T x B), nP x T x B
            keys,values after corruption

        """
        # the 1st part is the encoding phase
        # all remaining parts are query phase
        n_parts = len(o_keys_vec)
        # get a list of p_rm, only the 1st phase is the encoding phase
        # the rest of phases are considered as recall phases
        p_rms = [p_rm_ob_enc] * (n_parts - 1) + [p_rm_ob_rcl]
        # zero out random rows (time steps)
        for ip in range(n_parts):
            # zero out both key and values
            if self.rm_kv:
                [o_keys_vec[ip], o_vals_vec[ip]] = _zero_out_random_rows(
                    [o_keys_vec[ip], o_vals_vec[ip]], p_rms[ip],
                    n_rm_fixed=self.n_rm_fixed
                )
            # zero out values only
            # in this case the agent know which state is unknown
            else:
                [o_vals_vec[ip]] = _zero_out_random_rows(
                    [o_vals_vec[ip]], p_rms[ip],
                    n_rm_fixed=self.n_rm_fixed
                )
        return o_keys_vec, o_vals_vec

    def _delay_pred_demand(self, o_sample_, q_sample_):
        """apply delay to the queries, and zero pad the end of observations

        Parameters
        ----------
        o_sample_ : list
            observations
        q_sample_ : list
            queries

        Returns
        -------
        list, list
            padded observations and queries

        """
        if self.pad_len == 0 or self.max_pad_len == 0:
            return o_sample_, q_sample_
        # uniformly sample a padding length
        if self.pad_len == 'random':
            # high is exclusive so need to add 1
            pad_len = np.random.randint(low=0, high=self.max_pad_len + 1)
        # fixed padding length
        elif self.pad_len > 0:
            pad_len = self.pad_len
        else:
            raise ValueError(f'Invalid delay length: {self.pad_len}')

        # padd the data
        o_sample_ = _zero_pad_kvc(o_sample_, pad_len, side='bot')
        q_sample_ = _zero_pad_kvc(q_sample_, pad_len, side='top')
        return o_sample_, q_sample_


def _zero_out_random_rows(matrices, p_rm, n_rm_fixed=True):
    """zero out the same set of (randomly selected) rows for all input matrices

    Parameters
    ----------
    matrices : list
        a list of 2d arrays
    p_rm : float
        probability for set a row of zero

    Returns
    -------
    list
        a list of 2d arrays
    """
    assert 0 <= p_rm <= 1
    n_rows, _ = np.shape(matrices[0])
    for matrix in matrices:
        assert np.shape(matrix)[0] == n_rows
    # select # row(s) to zero out
    if n_rm_fixed:
        n_rows_to0 = np.ceil(p_rm * n_rows)
    else:
        # in this case, p_rm == E[rows_to_remove]
        max_rows_to_remove = p_rm * n_rows
        n_rows_to0 = np.round(np.random.uniform(high=max_rows_to_remove))
    # select some rows to zero out
    rows_to0 = np.random.choice(
        range(n_rows), size=int(n_rows_to0), replace=False
    )
    # zero out the same rows for all input matrices
    for i in range(len(matrices)):
        matrices[i][rows_to0, :] = 0
    return matrices


def _zero_pad_kvc(kvc: list, pad_len: int, side: str):
    """delay the prediction demand by shifting the query value to later time
    points

    Parameters
    ----------
    kvc : list
        Description of parameter `kvc`.
    pad_len : int
        Description of parameter `pad_len`.

    Returns
    -------
    type
        Description of returned object.

    """
    # unpack data
    keys_vec, vals_vec, ctxs_vec = kvc
    n_parts, n_params, k_dim = np.shape(keys_vec)
    _, _, v_dim = np.shape(vals_vec)
    _, c_dim = np.shape(ctxs_vec)
    # pad to delay prediction time
    keys_vec = [_vpad(k_mat, pad_len, side=side) for k_mat in keys_vec]
    vals_vec = [_vpad(v_mat, pad_len, side=side) for v_mat in vals_vec]
    # TODO here i assumed context is always in sync with the queries
    # but probably want to generate additional context for the padding period
    ctxs_vec = _vpad(ctxs_vec, pad_len, side='top')
    # pack the data
    kvc_ = [keys_vec, vals_vec, ctxs_vec]
    return kvc_


def _vpad(matrix, pad_len: int, side: str):
    '''vertically pad zeros from the top or bot'''
    #
    n_rows, n_cols = np.shape(matrix)
    zero_padding = np.zeros((pad_len, n_cols))
    if side == 'top':
        padded_matrix = np.vstack([zero_padding, matrix])
    elif side == 'bot':
        padded_matrix = np.vstack([matrix, zero_padding])
    else:
        raise ValueError('Unrecognizable padding side')
    return padded_matrix


'''test'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # from task.utils import sample_rand_path,sample_def_tps
    # init a graph
    n_param, n_branch = 16, 4
    n_parts = 2
    pad_len = 0
    # def_tps = np.array([1, 0] * 5)
    # def_tps = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # def_tps = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def_tps = np.array([1, 0] * (n_param // 2))
    def_tps = np.array([0, 1] * (n_param // 2))
    # def_path = np.tile(np.array([[1, 0], [0, 1]]), (1, 5)).T
    # def_path = np.tile(np.array([[1, 0], [0, 1]]), (1, n_param // 2)).T
    def_path = np.vstack([[1, 0] for i in range(n_param)])
    def_prob = .9
    # pad_len = 'random'
    p_rm_ob_enc, p_rm_ob_rcl = .5, .5
    p_rm_ob_enc, p_rm_ob_rcl = 0, 0
    key_rep_type = 'time'
    permute_queries = False
    permute_observations = False
    # key_rep_type = 'gaussian'
    sampler = StimSampler(
        n_param, n_branch,
        pad_len=pad_len,
        key_rep_type=key_rep_type,
        def_tps=def_tps, def_path=def_path, def_prob=def_prob
    )
    sample_, misc = sampler.sample(
        n_parts, p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl,
        # permute_queries=permute_queries, permute_observations=permute_observations
    )
    observations, queries = sample_
    [o_keys_vec, o_vals_vec, o_ctxs_vec] = observations
    [q_keys_vec, q_vals_vec, q_ctxs_vec] = queries

    # plot
    cmap = 'bone'
    n_timesteps = n_param
    width_ratios = [sampler.k_dim, sampler.v_dim] * 2 + [sampler.c_dim]
    f, axes = plt.subplots(
        n_parts, 5, figsize=(8, 5), sharey=True,
        gridspec_kw={'width_ratios': width_ratios}
    )
    for ip in range(n_parts):
        axes[ip, 0].imshow(o_keys_vec[ip], cmap=cmap)
        axes[ip, 1].imshow(o_vals_vec[ip], cmap=cmap)
        axes[ip, 2].imshow(q_keys_vec[ip], cmap=cmap)
        axes[ip, 3].imshow(q_vals_vec[ip], cmap=cmap)

    axes[0, 4].imshow(o_ctxs_vec, cmap=cmap)
    axes[1, 4].imshow(q_ctxs_vec, cmap=cmap)

    # label
    axes[0, 0].set_title('Observation')
    axes[0, 2].set_title('Queries')
    axes[-1, 0].set_xlabel('Keys/States')
    axes[-1, 1].set_xlabel('Values/Action')
    axes[-1, 2].set_xlabel('Keys/States')
    axes[-1, 3].set_xlabel('Values/Action')
    axes[0, 4].set_title('o, Context')
    axes[1, 4].set_title('q, Context')
    # modify y ticks/labels
    for ip in range(n_parts):
        axes[ip, 0].set_yticks(range(n_timesteps))
        axes[ip, 0].set_yticklabels(range(n_timesteps))
        axes[ip, 0].set_ylabel(f'Time, part {ip+1}')
    f.tight_layout()
