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
            key_rep='node',
            sampling_mode='enumerative'
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        self._form_representation(key_rep=key_rep)
        self.event_schema = Schema(
            n_param=n_param,
            n_branch=n_branch,
            key_rep=key_rep,
            sampling_mode=sampling_mode
        )

    def _form_representation(self, key_rep):
        # build state space and action space
        if key_rep == 'node':
            self.k_dim = self.n_param * self.n_branch
            self.v_dim = self.n_branch
        elif key_rep == 'time':
            self.k_dim = self.n_param
            self.v_dim = self.n_branch
        else:
            raise ValueError(f'unrecog representation type {key_rep}')
        # form representation
        self.key_rep = np.eye(self.k_dim)
        self.val_rep = np.eye(self.v_dim)

    def _sample(self):
        """sample an event sequence, one-hot vector representation

        Returns
        -------
        2d np array, 2d np array; T x (T x B), T x B
            sequence of keys / parameter values over time

        """
        # sample keys and parameter values, integer representation
        keys, vals = self.event_schema.sample()
        # translate to vector representation
        keys_vec = np.vstack([self.key_rep[k_t, :] for k_t in keys])
        vals_vec = np.vstack([self.val_rep[v_t, :] for v_t in vals])
        return keys_vec, vals_vec

    def sample(
            self,
            n_parts=1,
            p_rm_ob_enc=0,
            p_rm_ob_rcl=0,
            permute_queries=False,
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
        keys_vec_, vals_vec_ = self._sample()
        # sample for the observation phase
        o_keys_vec, o_vals_vec = self._sample_permutations(
            keys_vec_, vals_vec_, n_parts)
        # sample for the query phase
        if permute_queries:
            q_keys_vec, q_vals_vec = self._sample_permutations(
                keys_vec_, vals_vec_, n_parts)
        else:
            q_keys_vec = np.stack([keys_vec_ for _ in range(n_parts)])
            q_vals_vec = np.stack([vals_vec_ for _ in range(n_parts)])
        # corrupt input during encoding
        o_keys_vec, o_vals_vec = self._corrupt_observations(
            o_keys_vec, o_vals_vec, p_rm_ob_enc, p_rm_ob_rcl)
        # pack sample
        sample_ = [o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec]
        return sample_

    def _sample_permutations(self, keys_vec_raw, vals_vec_raw, n_perms):
        """given some raw key-val pairs, generate temporal permutation sets
        """
        T = self.n_param
        keys_vec = np.zeros((n_perms, T, self.k_dim))
        vals_vec = np.zeros((n_perms, T, self.n_branch))
        for ip in range(n_perms):
            # unique permutation for each movie part
            perm_op = np.random.permutation(T)
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


# '''test'''
#
# # init a graph
# n_param, n_branch = 3, 2
# n_parts = 2
# p_rm_ob_enc, p_rm_ob_rcl = .25, 0
#
# sampler = StimSampler(n_param, n_branch, key_rep='node')
# sample_ = sampler.sample(
#     n_parts, p_rm_ob_enc, p_rm_ob_rcl
# )
# [o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec] = sample_
#
# # plot
# cmap = 'bone'
# n_timesteps = n_param
# width_ratios = [sampler.k_dim, sampler.v_dim]*2
# f, axes = plt.subplots(
#     n_parts, 4, figsize=(9, 4), sharey=True,
#     gridspec_kw={'width_ratios': width_ratios}
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
