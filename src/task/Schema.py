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
        self._form_representation(key_rep_type)
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

    def _form_representation(self, key_rep_type):
        # build state space and action space
        if key_rep_type == 'node':
            self.k_dim = self.n_param * self.n_branch
            self.v_dim = self.n_branch
        elif key_rep_type == 'time':
            self.k_dim = self.n_param
            self.v_dim = self.n_branch
        else:
            raise ValueError(f'unrecog representation type {key_rep_type}')
        # form representation
        self.key_rep = np.eye(self.k_dim)
        self.val_rep = np.eye(self.v_dim)

# # key_used_ = set()
# #
# # T = 3
# # key_all = set(range(T))
# '''tests'''
#
# # init a graph
# n_param, n_branch = 6, 2
# schema = Schema(n_param, n_branch, key_rep_type='time')
# schema.key_rep_type
# key, val = schema.sample()
# print(key)
# print(val)
# np.shape(schema.transition_matrix)
# print(schema.transition_matrix)
# schema.transition_matrix[0, :]


# np.random.choice(range(n_branch), 10).astype(np.int16)
