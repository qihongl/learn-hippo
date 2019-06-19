'''
a generative model of event sequences, represented by one-hot vectors
samples from this class are processed to NN-readable form
'''

import numpy as np
# import matplotlib.pyplot as plt
from task.Schema import Schema


class StimSampler():
    def __init__(
            self,
            n_param, n_branch,
            sampling_mode='enumerative'
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        self.dim = n_branch * n_param
        # build state space and action space
        self.key_set = np.eye(n_param * n_branch)
        self.val_set = np.eye(n_branch)
        self.event_schema = Schema(
            n_param=n_param,
            n_branch=n_branch,
            sampling_mode=sampling_mode
        )

    def _sample(self, n_timestep):
        """sample an event sequence, one-hot vector representation

        Parameters
        ----------
        n_timestep : int
            the number of time steps of an "event part"

        Returns
        -------
        2d np array, 2d np array; T x (T x B), T x B
            sequence of keys / parameter values over time

        """
        # sample keys and parameter values, integer representation
        keys, vals = self.event_schema.sample(n_timestep)
        # convert to one-hot vector representation
        keys_vec, vals_vec = self.to_one_hot(keys, vals)
        return keys_vec, vals_vec

    def to_one_hot(self, keys, vals):
        # convert to one-hot vector representation
        keys_vec = np.vstack([self.key_set[s_t, :] for s_t in keys])
        vals_vec = np.vstack([self.val_set[p_t, :] for p_t in vals])
        return keys_vec, vals_vec

    def sample(
            self, n_timestep,
            n_parts=1,
            p_rm_ob_enc=0, p_rm_ob_rcl=0,
            permute_queries=False,
            xy_format=True, stack=True,
    ):
        """sample a multi-part "movie", with repetition structure

        Parameters
        ----------
        n_timestep : int
            the number of time steps of EACH PART
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
        keys_vec_, vals_vec_ = self._sample(n_timestep)
        # sample for the observation phase
        o_keys_vec, o_vals_vec = self._sample_permutations(
            keys_vec_, vals_vec_, n_timestep, n_parts)
        # sample for the query phase
        if permute_queries:
            q_keys_vec, q_vals_vec = self._sample_permutations(
                keys_vec_, vals_vec_, n_timestep, n_parts)
        else:
            q_keys_vec = np.stack([keys_vec_ for _ in range(n_parts)])
            q_vals_vec = np.stack([vals_vec_ for _ in range(n_parts)])
        # corrupt input during encoding
        o_keys_vec, o_vals_vec = self._corrupt_observations(
            o_keys_vec, o_vals_vec, p_rm_ob_enc, p_rm_ob_rcl)
        # pack sample
        sample_ = [o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec]
        if xy_format:
            # return RNN readable form
            return _to_xy(sample_, stack=stack)
        # return human readable form
        return sample_

    def _sample_permutations(
            self,
            keys_vec_raw, vals_vec_raw,
            n_timestep, n_perms
    ):
        """given some raw key-val pairs, generate temporal permutation sets
        """
        keys_vec = np.zeros((n_perms, n_timestep, self.dim))
        vals_vec = np.zeros((n_perms, n_timestep, self.n_branch))
        for ip in range(n_perms):
            # unique permutation for each movie part
            perm_op = np.random.permutation(n_timestep)
            keys_vec[ip] = keys_vec_raw[perm_op, :]
            vals_vec[ip] = vals_vec_raw[perm_op, :]
        return keys_vec, vals_vec

    def _corrupt_observations(
        self,
        o_keys_vec, o_vals_vec,
        p_rm_ob_enc, p_rm_ob_rcl
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
        p_rms = [p_rm_ob_enc] + [p_rm_ob_rcl] * (n_parts-1)
        # zero out random rows (time steps)
        for ip in range(n_parts):
            [o_keys_vec[ip], o_vals_vec[ip]] = _zero_out_random_rows(
                [o_keys_vec[ip], o_vals_vec[ip]], p_rms[ip])
        return o_keys_vec, o_vals_vec


def _to_xy(sample_, stack=True):
    """to RNN readable form, where x/y is the input/target sequence

    Parameters
    ----------
    sample_ : [list, list]
        the output of self.sample

    Returns
    -------
    2d array, 2d array
        the input/target sequence to the RNN

    """
    [o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec] = sample_
    n_parts = len(o_keys_vec)
    x = [None] * n_parts
    y = [None] * n_parts
    for ip in range(n_parts):
        x[ip] = np.hstack([o_keys_vec[ip], o_vals_vec[ip], q_keys_vec[ip]])
        y[ip] = np.hstack([q_vals_vec[ip]])
    if stack:
        x = np.vstack([x[ip] for ip in range(n_parts)])
        y = np.vstack([y[ip] for ip in range(n_parts)])
    return x, y


def _zero_out_random_rows(matrices, p_rm):
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
    n_rows_to0 = int(np.ceil(p_rm * n_rows))
    rows_to0 = np.random.choice(range(n_rows), size=n_rows_to0, replace=False)
    for i in range(len(matrices)):
        matrices[i][rows_to0, :] = 0
    return matrices

# keys_vec_, vals_vec_ = sampler._sample(n_timesteps)
#
# q_keys_vec_ = np.stack([keys_vec_ for _ in range(n_parts)])
#
# np.stack([keys_vec_ for _ in range(n_parts)])

#
# '''test'''
#
# # init a graph
# n_param, n_branch = 3, 2
# n_timesteps = n_param
# n_parts = 2
# p_rm_ob_enc, p_rm_ob_rcl = .25, 0
#
#
# sampler = StimSampler(n_param, n_branch)
# sample_ = sampler.sample(
#     n_timesteps, n_parts, p_rm_ob_enc, p_rm_ob_rcl,
#     xy_format=False, stack=True
# )
# [o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec] = sample_
#
# # plot
# cmap = 'bone'
# rk, rv = n_param * n_branch, n_branch
# f, axes = plt.subplots(
#     n_parts, 4, figsize=(9, 4), sharey=True,
#     gridspec_kw={'width_ratios': [rk, rv, rk, rv]}
# )
# for ip in range(n_parts):
#     axes[ip, 0].imshow(o_keys_vec[ip], cmap=cmap)
#     axes[ip, 1].imshow(o_vals_vec[ip], cmap=cmap)
# for ip in range(n_parts):
#     axes[ip, 2].imshow(q_keys_vec[ip], cmap=cmap)
#     axes[ip, 3].imshow(q_vals_vec[ip], cmap=cmap)
# # label
# # axes[0, 0].set_title('Observation')
# # axes[0, 2].set_title('Queries')
# axes[-1, 0].set_xlabel('Keys/States')
# axes[-1, 1].set_xlabel('Values/Action')
# axes[-1, 2].set_xlabel('Keys/States')
# axes[-1, 3].set_xlabel('Values/Action')
# # modify y ticks/labels
# for ip in range(n_parts):
#     axes[ip, 0].set_yticks(range(n_timesteps))
#     axes[ip, 0].set_yticklabels(range(n_timesteps))
#     axes[ip, 0].set_ylabel(f'Time, part {ip+1}')
# f.tight_layout()
#
# # # x, y = sample_
# # x, y = _to_xy(sample_)
# # f, axes = plt.subplots(
# #     1, 2, figsize=(9, 4), sharey=True,
# #     gridspec_kw={'width_ratios': [rk+rv+rk, rv]}
# # )
# # axes[0].imshow(x, cmap=cmap)
# # axes[1].imshow(y, cmap=cmap)
