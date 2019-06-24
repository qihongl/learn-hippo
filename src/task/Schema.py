import numpy as np
# import matplotlib.pyplot as plt

VALID_SAMPLING_MODE = ['enumerative']
KEY_REPRESENTATION = ['node', 'time']
# KEY_REPRESENTATION = ['node', 'time', 'gaussian']
# VALID_SAMPLING_MODE = ['enumerative', 'probabilistic']
# TODO: sample w.r.t to transition matrix
# TODO: implement probabilistic sampling mode


class Schema():
    '''
    a generative model of sequences
    - has integer representation of key and values
    - and corresponding representation
    '''

    def __init__(
            self, n_param, n_branch,
            context_onehot=True,
            context_dim=1,
            context_drift=False,
            key_rep_type='node',
            sampling_mode='enumerative'
            # def_path=None, def_prob=None,
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        # self.def_prob = def_prob
        # self.def_path = def_path
        # sampling mode
        self.key_rep_type = key_rep_type
        self.sampling_mode = sampling_mode
        self._form_key_val_representation(key_rep_type)
        self._form_context_representation(
            context_onehot, context_drift, context_dim
        )
        assert key_rep_type in KEY_REPRESENTATION
        assert sampling_mode in VALID_SAMPLING_MODE

    def sample(self):
        """sample an event sequence, integer representation

        Returns
        -------
        1d np array, 1d np array; T x 1, T x 1
            sequence of key / parameter values over time

        """
        key = self._sample_key()
        val = np.random.choice(
            range(self.n_branch), size=self.n_param, replace=True
        ).astype(np.int16)
        # context id and values are consistent
        return key, val

    def _sample_key(self):
        T = self.n_param
        if self.key_rep_type == 'node':
            key_branch_id = np.array([
                np.random.choice(range(self.n_branch)) for _ in range(T)
            ])
            time_shifts = np.array([self.n_branch * t for t in range(T)])
            key = key_branch_id + time_shifts
        elif self.key_rep_type == 'time':
            # key = np.random.permutation(T)
            key = np.arange(T)
        else:
            raise ValueError(f'unrecog representation type {self.key_rep_type}')
        return key.astype(np.int16)

    def _form_key_val_representation(self, key_rep_type):
        # build state space and action space
        if key_rep_type == 'node':
            self.key_rep = np.eye(self.n_param * self.n_branch)
            self.val_rep = np.eye(self.n_branch)
        elif key_rep_type == 'time':
            self.key_rep = np.eye(self.n_param)
            self.val_rep = np.eye(self.n_branch)
        else:
            raise ValueError(f'unrecog representation type {key_rep_type}')

        # get dimension
        self.k_dim = np.shape(self.key_rep)[1]
        self.v_dim = np.shape(self.val_rep)[1]

    def _form_context_representation(
            self,
            context_onehot, context_drift, context_dim
    ):
        self.context_onehot = context_onehot
        self.context_drift = context_drift
        if context_onehot:
            self.c_dim = self.n_param
        else:
            self.c_dim = context_dim
        # form context representation
        if self.context_onehot:
            self.ctx_rep = np.eye(self.n_param)
        else:
            norm_heuristic = 2
            self.ctx_rep = sample_context_drift(
                self.c_dim, self.n_param,
                norm=norm_heuristic,
                dynamic=self.context_drift
            )


def sample_context_drift(
        n_dim, n_point,
        norm=1,
        end_scale=1,
        noise_scale=.01,
        normalize=True,
        normalizer=1,
        dynamic=True,
):
    """sample n_dim random walk

    Parameters
    ----------
    n_dim : type
        Description of parameter `n_dim`.
    n_point : type
        Description of parameter `n_point`.
    end_loc : type
        Description of parameter `end_loc`.
    end_scale : type
        Description of parameter `end_scale`.
    noise_scale : type
        Description of parameter `noise_scale`.
    normalize : type
        Description of parameter `normalize`.

    Returns
    -------
    type
        Description of returned object.

    """
    end_point = np.random.normal(
        loc=np.random.normal(size=(n_dim,)),
        scale=end_scale, size=(n_dim,))
    # normalize the context by some metric
    if normalize:
        end_point /= np.linalg.norm(end_point, ord=normalizer)
    # set the norm
    end_point *= norm
    # decide if the context is drifting or fixed
    if dynamic:
        # convec interpolation
        ws = np.linspace(0, 1, n_point)
        path = np.array([w * end_point for w in ws])
    else:
        # copy t times
        path = np.tile(end_point, (n_point, 1))
    if noise_scale > 0:
        path += np.random.normal(scale=noise_scale, size=np.shape(path))
    return path


# '''tests'''
#
# # init a graph
# n_param, n_branch = 6, 2
# schema = Schema(
#     n_param, n_branch,
#     context_dim=1,
#     key_rep_type='time'
# )
# schema.key_rep_type
# key, val = schema.sample()
# print(key)
# print(val)
# # np.shape(schema.transition_matrix)
# # print(schema.transition_matrix)
# # schema.transition_matrix[0, :]
# # np.shape(np.zeros((4, 3)))
# # np.random.choice(range(n_branch), 10).astype(np.int16)
#
# # np.shape()
# #
# n_timesteps = 10
# context_dim = 2
# #
# # schema.ctx_rep
# P = sample_context_drift(context_dim, n_timesteps)
# p = P[-1, :]
#
# np.tile(p, (n_timesteps, 1))
