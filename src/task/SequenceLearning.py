import numpy as np
from collections import deque
from utils.utils import to_pth
from task.utils import get_event_ends
from analysis.task import compute_event_similarity
from task.StimSampler import StimSampler
# import pdb
# pdb.set_trace()


class SequenceLearning():
    """a key-value assoc learning task with explicit query keys input,
    - ... where query keys are not explicitly presented
    - but queries are always ordered by "time", so the model knows which
    element is being queried
    """

    def __init__(
            self,
            n_param,
            n_branch,
            pad_len=0,
            max_pad_len=None,
            n_parts=2,
            def_path=None,
            def_prob=None,
            def_tps=None,
            p_rm_ob_enc=0,
            p_rm_ob_rcl=0,
            n_rm_fixed=False,
            similarity_max=None,
            similarity_min=None,
            similarity_cap_lag=2,
            permute_queries=False,
            permute_observations=True,
            key_rep_type='time',
            sampling_mode='enumerative',
            repeat_query=False,
    ):
        # build a sampler
        self.stim_sampler = StimSampler(
            n_param=n_param,
            n_branch=n_branch,
            pad_len=pad_len,
            max_pad_len=max_pad_len,
            def_path=def_path,
            def_prob=def_prob,
            def_tps=def_tps,
            key_rep_type=key_rep_type,
            n_rm_fixed=n_rm_fixed,
            sampling_mode=sampling_mode,
            repeat_query=repeat_query,
        )
        # graph param
        self.n_param = n_param
        self.n_branch = n_branch
        self.n_parts = n_parts
        self.pad_len = pad_len
        #
        self.max_pad_len = self.stim_sampler.max_pad_len
        self.T_part_max = self.n_param + self.max_pad_len
        self.T_total_max = self.T_part_max * self.n_parts
        # "noise" in the obseravtion
        self.p_rm_ob_enc = p_rm_ob_enc
        self.p_rm_ob_rcl = p_rm_ob_rcl
        self.n_rm_fixed = n_rm_fixed
        # whether to permute queries
        self.permute_queries = permute_queries
        self.permute_observations = permute_observations
        # task dimension
        self.k_dim = self.stim_sampler.k_dim
        self.v_dim = self.stim_sampler.v_dim
        self.x_dim = self.k_dim * 2 + self.v_dim
        self.y_dim = self.v_dim
        # stats
        # expected inter event similarity under uniform assumption
        self.similarity_cap_lag = similarity_cap_lag
        if similarity_max is None:
            self.similarity_max = (n_branch - 1) / n_branch
        else:
            self.similarity_max = similarity_max
        if similarity_min is None:
            self.similarity_min = 0
        else:
            self.similarity_min = similarity_min

    def sample(
            self, n_samples,
            interleave=False, to_torch=True, return_misc=False
    ):
        # prealloc, agnostic about sequence length
        X = [None] * n_samples
        Y = [None] * n_samples
        misc = [None] * n_samples
        # track the last k events
        prev_events = deque(maxlen=self.similarity_cap_lag)

        # generate samples
        i = 0
        while i < n_samples:
            sample_i, misc_i = self.stim_sampler.sample(
                n_parts=self.n_parts,
                p_rm_ob_enc=self.p_rm_ob_enc,
                p_rm_ob_rcl=self.p_rm_ob_rcl,
                permute_queries=self.permute_queries,
                permute_observations=self.permute_observations,
            )
            # compute similarity(event_i vs. event_j) for j in prev-k-events
            _, Y_i_int = misc_i
            prev_sims = np.array([compute_event_similarity(Y_j_int, Y_i_int)
                                  for Y_j_int in prev_events])
            if np.any(prev_sims > self.similarity_max) or np.any(prev_sims < self.similarity_min):
                continue
            else:
                # collect data
                prev_events.append(Y_i_int)
                misc[i] = misc_i
                X[i], Y[i] = _to_xy(sample_i)
                i += 1

        if interleave:
            X, Y = interleave_stories(X, Y, self.n_parts)
            n_samples = n_samples // 2

        # type conversion
        if to_torch:
            X = [to_pth(X[i]) for i in range(n_samples)]
            Y = [to_pth(Y[i]) for i in range(n_samples)]
        if return_misc:
            return X, Y, misc
        return X, Y

    def get_time_param(self, T_total):
        """compute time related parameters
        since it might be unique for each example

        Parameters
        ----------
        T_total : int
            the 0-th dim of X_i or Y_i, i.e. the input sequnce length
            T_total = (n_param + pad_len) x n_parts

        Returns
        -------
        type
            Description of returned object.

        """
        T_part = T_total // self.n_parts
        pad_len = T_part - self.n_param
        event_ends = get_event_ends(T_part, self.n_parts)
        event_bonds = [event_ends[i] + 1 for i in range(len(event_ends) - 1)]
        return T_part, pad_len, event_ends, event_bonds

    def get_pred_time_mask(self, T_total, T_part, pad_len, dtype=bool):
        """get a mask s.t.
        mask[t] == False => during prediction delay period, no prediction demands
        mask[t] == True  => prediction time step

        Parameters
        ----------
        T_total : int
            total duration of the event sequence
        T_part : int
            duration of one part
        pad_len : int
            padding length, duration of the delay
        dtype : type
            data type

        Returns
        -------
        1d array
            the mask

        """
        return np.array([t % T_part >= pad_len for t in range(T_total)], dtype=dtype)


def _to_xy(sample_):
    # unpack data
    observations, queries = sample_
    [o_keys_vec, o_vals_vec, o_ctxs_vec] = observations
    [q_keys_vec, q_vals_vec, q_ctxs_vec] = queries
    # form x and y
    x = np.hstack([
        np.vstack([k for k in o_keys_vec]),
        np.vstack([v for v in o_vals_vec]),
        np.vstack([k for k in q_keys_vec]),
    ])
    y = np.vstack(q_vals_vec)
    return x, y


def _split_xy(X_, Y_, n_parts):
    X_split_ = np.array_split(X_, n_parts, axis=0)
    Y_split_ = np.array_split(Y_, n_parts, axis=0)
    return X_split_, Y_split_


def _interleave_ab(array_a, array_b):
    return [ab for pair in zip(array_a, array_b) for ab in pair]


def interleave_stories(X, Y, n_parts):
    n_stories = len(Y)
    assert n_stories % 2 == 0
    X_ab, Y_ab = [], []
    # loop over all 2-samples pairs
    for i in np.arange(0, n_stories, 2):
        # get story a and story b
        a, b = i, i + 1
        # get sub-sequences for a and b
        X_a_split, Y_a_split = _split_xy(X[a], Y[a], n_parts)
        X_b_split, Y_b_split = _split_xy(X[b], Y[b], n_parts)
        # interleave them
        X_ab_ = np.vstack(_interleave_ab(X_a_split, X_b_split))
        Y_ab_ = np.vstack(_interleave_ab(Y_a_split, Y_b_split))
        # collect data
        X_ab.append(X_ab_)
        Y_ab.append(Y_ab_)
    return X_ab, Y_ab


'''how to use'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from task.utils import scramble_array

    n_param, n_branch = 6, 3
    n_parts = 2
    # p_rm_ob_enc = 0.5
    # p_rm_ob_rcl = 0.5
    p_rm_ob_enc = 0
    p_rm_ob_rcl = 0
    similarity_max = .5
    similarity_min = 1 / n_branch
    permute_queries = False
    permute_observations = False
    # pad_len = 'random'
    pad_len = 0
    task = SequenceLearning(
        n_param=n_param, n_branch=n_branch, pad_len=pad_len,
        p_rm_ob_enc=p_rm_ob_enc, p_rm_ob_rcl=p_rm_ob_rcl, n_parts=n_parts,
        similarity_max=similarity_max, similarity_min=similarity_min,
        permute_queries=permute_queries, permute_observations=permute_observations
    )

    # gen samples
    n_samples = 10
    X, Y, misc = task.sample(n_samples, to_torch=False, return_misc=True)

    # get a sample
    i = 0
    X_i, Y_i = X[i], Y[i]
    np.shape(X_i)
    np.shape(Y_i)
    # # option 1: scramble observations
    # X_i[:, :task.k_dim + task.v_dim] = scramble_array(
    #     X_i[:, :task.k_dim+task.v_dim])
    # # option 2: scramble observations + queries
    # [X_i, Y_i] = scramble_array_list([X_i, Y_i])

    # compute time info
    T_total = np.shape(X_i)[0]
    T_part, pad_len, event_ends, event_bonds = task.get_time_param(T_total)

    # plot
    cmap = 'bone'
    f, axes = plt.subplots(
        1, 2, figsize=(6, 6),
        gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]}
    )
    axes[0].imshow(X_i, cmap=cmap, vmin=0, vmax=1)
    axes[1].imshow(Y_i, cmap=cmap, vmin=0, vmax=1)

    for ax in axes:
        for event_bond in event_bonds:
            ax.axhline(event_bond - .5, color='red', linestyle='--')
    axes[0].axvline(task.k_dim - .5, color='red', linestyle='--')
    axes[0].axvline(task.k_dim + task.v_dim - .5, color='red', linestyle='--')

    # '''interleaved story'''
    # X, Y, misc = task.sample(
    #     n_samples, interleave=True, to_torch=False, return_misc=True
    # )
    # # get a sample
    # i = 0
    # X_ab, Y_ab = X[i], Y[i]
    # # X_ab, Y_ab = interleave_stories(X, Y, n_parts)
    # # X_ab, Y_ab = X_ab[0], Y_ab[0]
    #
    # cmap = 'bone'
    # f, axes = plt.subplots(
    #     1, 2, figsize=(6, 12),
    #     gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]}
    # )
    # axes[0].imshow(X_ab, cmap=cmap, vmin=0, vmax=1)
    # axes[1].imshow(Y_ab, cmap=cmap, vmin=0, vmax=1)
    #
    # T_total = np.shape(Y_ab)[0]
    # for eb in np.arange(0, T_total, n_param)[1:]:
    #     for ax in axes:
    #         ax.axhline(eb - .5, color='red', linestyle='--')
    # axes[0].axvline(task.k_dim - .5, color='red', linestyle='--')
    # axes[0].axvline(task.k_dim + task.v_dim - .5, color='red', linestyle='--')
    # axes[0].set_xlabel('o-key | o-val | q-key')
    # axes[1].set_xlabel('q-val')
    #
    # yticks = [eb - n_param //
    #           2 for eb in np.arange(0, T_total + 1, n_param)[1:]]
    # yticklabels = ['A', 'B'] * n_parts
    # axes[0].set_yticks(yticks)
    # axes[0].set_yticklabels(yticklabels)
