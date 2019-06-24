import numpy as np
from utils.utils import to_pth
from task.StimSampler import StimSampler
from utils.constants import RNR_COND_DICT
from itertools import combinations
from scipy.misc import comb
# import matplotlib.pyplot as plt


class RNR():
    """the recall/no-recall task
    """

    def __init__(
            self,
            n_param,
            n_branch,
            n_parts=3,
            p_rm_ob_enc=0,
            p_rm_ob_rcl=0,
            n_rm_fixed=True,
            context_onehot=True,
            context_dim=1,
            context_drift=False,
            append_context=False,
            key_rep_type='time',
            sampling_mode='enumerative'
    ):
        # build a sampler
        self.stim_sampler = StimSampler(
            n_param, n_branch,
            context_onehot=context_onehot,
            context_dim=context_dim,
            key_rep_type=key_rep_type,
            n_rm_fixed=n_rm_fixed,
            sampling_mode=sampling_mode,
            context_drift=context_drift
        )
        # graph param
        self.n_param = n_param
        self.n_branch = n_branch
        # "noise" in the obseravtion
        self.p_rm_ob_enc = p_rm_ob_enc
        self.p_rm_ob_rcl = p_rm_ob_rcl
        self.n_rm_fixed = n_rm_fixed
        #
        self.append_context = append_context
        self.context_drift = context_drift
        # task duration
        self.T_part = n_param
        self.n_parts = n_parts
        self.T_total = self.T_part * n_parts
        #
        self.n_conds = int(len(RNR_COND_DICT))
        self.batch_size = int(comb(self.n_parts, self.n_parts-1) * self.n_conds)
        # task dimension
        self.k_dim = self.stim_sampler.k_dim
        self.v_dim = self.stim_sampler.v_dim
        self.x_dim = self.k_dim + self.v_dim
        self.y_dim = self.v_dim
        if append_context:
            # augment x_dim
            self.c_dim = self.stim_sampler.c_dim
            self.x_dim += self.c_dim
        # input validation
        self._class_config_validation()

    def _class_config_validation(self):
        assert self.n_parts >= 3, 'n_parts >= 3'

    def sample(self, n_sample=1, stack=False, to_torch=True):
        # compute batch size
        n_total = self.batch_size * n_sample
        # prealloc
        if stack:
            X = np.zeros((n_total, self.T_total, self.x_dim))
            Y = np.zeros((n_total, self.T_total, self.y_dim))
        else:
            X = np.zeros((n_total, self.n_parts, self.T_part, self.x_dim))
            Y = np.zeros((n_total, self.n_parts, self.T_part, self.y_dim))
        mem_id = np.zeros(n_total,)
        cond_name = np.zeros(n_total,)
        # sample data
        for i in range(n_sample):
            ii = np.arange(self.batch_size) + i * self.batch_size
            data_batch_ = self._make_rnr_batch(stack)
            x_batch, y_batch, mem_id_batch, cond_id_batch = data_batch_
            # gather data
            X[ii] = x_batch
            Y[ii] = y_batch
            mem_id[ii] = mem_id_batch
            cond_name[ii] = cond_name
        # to tensor
        if to_torch:
            X = to_pth(X)
            Y = to_pth(Y)
        # pack metadata
        metadata = [mem_id, cond_name]
        return X, Y, metadata

    def _make_rnr_batch(self, stack=False):
        # 1. get ingredients for a RNR trial
        # for every event type, get 2 parts (for encoding vs. recall)
        n_parts_inner = 2
        # generate n samples
        # where n-1 is the number of competing memories during recall
        x_pts, y_pts = [None] * self.n_parts, [None] * self.n_parts
        for i in range(self.n_parts):
            # generate one sample
            self.stim_sampler.reset_schema()
            sample_i = self.stim_sampler.sample()
            # unpack data
            observations_i, queries_i = sample_i
            [o_keys_vec_i, o_vals_vec_i, o_ctxs_vec_i] = observations_i
            [q_keys_vec_i, q_vals_vec_i, q_ctxs_vec_i] = queries_i
            # form xy for the two parts
            x_pts[i] = [
                np.hstack([o_keys_vec_i[ip], o_vals_vec_i[ip], o_ctxs_vec_i])
                for ip in range(n_parts_inner)
            ]
            y_pts[i] = [q_vals_vec_i[ip] for ip in range(n_parts_inner)]

        # 2. form a RNR trial
        x_batch, y_batch = [], []
        rcl_mv_id_batch, cond_id_batch = [], []
        # for the same set of movies, create training data for all trial type
        for cond_id, cond_name in RNR_COND_DICT.items():
            print(cond_id, cond_name)
            # get encoding movies
            for enc_mv_ids in combinations(range(self.n_parts), self.n_parts-1):
                # compute the recall phase movie id depends on the condition
                if cond_name == 'NR':
                    rcl_mv_id = list(
                        set(range(self.n_parts)).difference(enc_mv_ids))[0]
                elif cond_name == 'R':
                    rcl_mv_id = np.random.choice(enc_mv_ids)
                else:
                    raise ValueError(
                        f'Unrecognizable RNR condition: {cond_name}')
                # choose k out of K movies as the encoding movies
                # print(enc_mv_ids)
                # collect data for the encoding phase ...
                x_enc = [x_pts[i][0] for i in enc_mv_ids]
                y_enc = [y_pts[i][0] for i in enc_mv_ids]
                # ... and the recall phase
                x_rcl = [x_pts[rcl_mv_id][1]]
                y_rcl = [y_pts[rcl_mv_id][1]]
                # combine the data for the encoding and recall phase
                x = np.vstack([x_enc, x_rcl])
                y = np.vstack([y_enc, y_rcl])
                # "stack out" the part structure
                if stack:
                    x = np.vstack([x[ip_] for ip_ in range(len(x))])
                    y = np.vstack([y[ip_] for ip_ in range(len(y))])

                # accumulate the batch
                x_batch.append(x)
                y_batch.append(y)
                rcl_mv_id_batch.append(rcl_mv_id)
                cond_id_batch.append(cond_id)

        return x_batch, y_batch, rcl_mv_id_batch, cond_id_batch


# '''testing'''
#
# n_param, n_branch = 4, 2
# n_parts = 3
# p_rm_ob_enc = 0
# p_rm_ob_rcl = 0
# n_samples = 5
# # context_dim = 10
# append_context = True
# task = RNR(
#     n_param, n_branch,
#     context_onehot=True,
#     # context_drift=True,
#     # context_dim=context_dim,
#     append_context=append_context,
# )
#
# data_batch_ = task._make_rnr_batch(stack=False)
# x_batch, y_batch, rcl_mv_id_batch, cond_id_batch = data_batch_
# np.shape(x_batch)
# i = 0
# x_i, y_i = x_batch[i], y_batch[i]
# rcl_mv_id_i, cond_id_i = rcl_mv_id_batch[i], cond_id_batch[i]
#
# n_parts_, n_timesteps_, x_dim = np.shape(x_i)
#
# cmap = 'bone'
# f, axes = plt.subplots(
#     3, 2, figsize=(7, 7), gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]})
#
# for ip in range(n_parts):
#     axes[ip, 0].imshow(x_i[ip], cmap=cmap)
#     axes[ip, 1].imshow(y_i[ip], cmap=cmap)
#     axes[ip, 0].set_ylabel(f'time, part {ip}')
#
# ox_label = 'key/val/ctx' if append_context else 'key/val'
# axes[-1, 0].set_xlabel(ox_label)
# axes[-1, 1].set_xlabel('val')
#
# f.suptitle(f'cond = {RNR_COND_DICT[cond_id_i]}, memory id = {rcl_mv_id_i}')
#
# for i in range(3):
#     print(np.arange(task.batch_size) + i * task.batch_size)
