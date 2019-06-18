'''
a generative model of event sequences, represented by integers
'''
import numpy as np
# from data.utils import sample_rand_path
# import matplotlib.pyplot as plt

VALID_SAMPLING_MODE = ['enumerative']
# VALID_SAMPLING_MODE = ['enumerative', 'probabilistic']
# TODO: sample w.r.t to transition matrix
# TODO: implement probabilistic sampling mode


class EventSchema():
    def __init__(
            self, n_param, n_branch,
            sampling_mode='enumerative'
            # def_path=None, def_prob=None,
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        # self.def_prob = def_prob
        # self.def_path = def_path
        # sampling mode
        self.sampling_mode = sampling_mode
        assert sampling_mode in VALID_SAMPLING_MODE

    def sample(self, n_timestep, sampling_mode='full'):
        """sample an event sequence, integer representation

        Parameters
        ----------
        n_timestep : int
            the number of time steps of this event sequence

        Returns
        -------
        1d np array, 1d np array; T x 1, T x 1
            sequence of states / parameter values over time

        """
        self._sample_input_validation(n_timestep)
        # prealloc
        states = np.zeros((n_timestep,), dtype=np.int16)
        param_vals = np.zeros((n_timestep,), dtype=np.int16)
        if self.sampling_mode == 'enumerative':
            for t in range(n_timestep):
                time_shift = self.n_branch * t
                states[t] = np.random.choice(range(self.n_branch))+time_shift
                param_vals[t] = np.random.choice(range(self.n_branch))
        return states, param_vals

    def _sample_input_validation(self, n_timestep):
        assert n_timestep <= self.n_param
        if self.sampling_mode == 'enumerative':
            assert n_timestep == self.n_param,\
                'n_timestep should = n_param when sampling_mode = `enumerative`'


# '''tests'''
#
# # init a graph
# n_param, n_branch = 7, 3
# n_timesteps = n_param
# es = EventSchema(n_param, n_branch)
# states, param_vals = es._sample(n_timesteps)
# states_vec, param_vals_vec = es.sample(n_timesteps)
#
# # np.shape(es.transition_matrix)
# # print(es.transition_matrix)
# # es.transition_matrix[0, :]
#
#
# # plt.imshow(states_vec)
# # plt.imshow(param_vals_vec)
