'''
a generative model of event sequences, represented by integers
'''
import numpy as np
# from data.utils import sample_rand_path
# import matplotlib.pyplot as plt

VALID_SAMPLING_MODE = ['enumerative']
KEY_REPRESENTATION = ['node', 'time']
# VALID_SAMPLING_MODE = ['enumerative', 'probabilistic']
# TODO: sample w.r.t to transition matrix
# TODO: implement probabilistic sampling mode


class Schema():
    def __init__(
            self, n_param, n_branch,
            key_rep='node',
            sampling_mode='enumerative'
            # def_path=None, def_prob=None,
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        # self.def_prob = def_prob
        # self.def_path = def_path
        # sampling mode
        self.key_rep = key_rep
        self.sampling_mode = sampling_mode
        assert key_rep in KEY_REPRESENTATION
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
        if self.key_rep == 'node':
            key_branch_id = np.array([
                np.random.choice(range(self.n_branch)) for _ in range(T)
            ])
            time_shifts = np.array([self.n_branch * t for t in range(T)])
            key = key_branch_id + time_shifts
        elif self.key_rep == 'time':
            key = np.random.permutation(T)
        else:
            raise ValueError(f'unrecog representation type {self.key_rep}')
        return key.astype(np.int16)


# # key_used_ = set()
# #
# # T = 3
# # key_all = set(range(T))
# '''tests'''
#
# # init a graph
# n_param, n_branch = 6, 2
# schema = Schema(n_param, n_branch, key_rep='time')
# schema.key_rep
# key, val = schema.sample()
# print(key)
# print(val)
# np.shape(schema.transition_matrix)
# print(schema.transition_matrix)
# schema.transition_matrix[0, :]


# np.random.choice(range(n_branch), 10).astype(np.int16)
