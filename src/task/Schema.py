import numpy as np

VALID_SAMPLING_MODE = ['enumerative']
KEY_REPRESENTATION = ['node', 'time']
# KEY_REPRESENTATION = ['node', 'time', 'gaussian']
# VALID_SAMPLING_MODE = ['enumerative', 'probabilistic']
# TODO: implement probabilistic sampling mode
# TODO: value sample for node-rep key is not general enough,
# all nodes at time t have the same next state transition


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
            def_path=None,
            def_prob=None,
            key_rep_type='node',
            sampling_mode='enumerative',
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        self.def_prob = def_prob
        self.def_path = def_path
        # sampling mode
        self.key_rep_type = key_rep_type
        self.sampling_mode = sampling_mode
        self._form_transition_matrix(def_path, def_prob)
        self._form_key_val_representation(key_rep_type)
        self._form_context_representation(
            context_onehot, context_drift, context_dim
        )
        assert key_rep_type in KEY_REPRESENTATION
        assert sampling_mode in VALID_SAMPLING_MODE

    def sample(self):
        """sample an event sequence

        Returns
        -------
        1d np array, 1d np array; T x 1, T x 1
            sequence of key / parameter values over time

        """
        return self._sample_key_val()

    def _sample_key_val(self):
        """sample a sequence of key-value pairs, which can be used to
        instantiate an event sequence

        if key_rep_type is ...
        "time": then key is one-hot representation of time
        "node": then key_t represents the state at time t

        Returns
        -------
        list, list
            keys, values

        """
        T = self.n_param
        # construct keys
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
        # sample values
        val = np.array([
            np.random.choice(range(self.n_branch), p=self.transition[t, :])
            for t in range(T)
        ])
        # type conversion
        val = val.astype(np.int16)
        key = key.astype(np.int16)
        return key, val

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

    def _form_transition_matrix(self, def_path=None, def_prob=None):
        """form the transition matrix (P x B) of the event schema graph

        Parameters
        ----------
        def_path : list/ 1d array
            the default/schematic path
        def_prob : float
            the probability of following the default path

        """
        # if the input graph params are un-specified, use uniform random graph
        if def_prob is None:
            def_prob = 1/self.n_branch
        if def_path is None:
            def_path = np.array([np.random.choice(range(self.n_branch))
                                 for _ in range(self.n_param)])
        # input validation
        assert 1/self.n_branch <= def_prob <= 1
        assert len(def_path) == self.n_param
        assert np.all(def_path < self.n_branch)
        def_path = def_path.astype(np.int16)

        # form the transition matrix
        self.transition = np.zeros((self.n_param, self.n_branch))
        for t in range(self.n_param):
            # assign p to the default node
            self.transition[t, def_path[t]] = def_prob
            # assign (1-p)/(B-1) to the rest
            non_def_prob = (1-def_prob) / (self.n_branch - 1)
            self.transition[t, self.transition[t, :] == 0] = non_def_prob


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
        # convex interpolation
        ws = np.linspace(0, 1, n_point)
        path = np.array([w * end_point for w in ws])
    else:
        # copy t times
        path = np.tile(end_point, (n_point, 1))
    if noise_scale > 0:
        path += np.random.normal(scale=noise_scale, size=np.shape(path))
    return path


'''tests'''
if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # init a graph
    n_param, n_branch = 6, 3
    def_prob = .5
    def_path = np.ones(n_param,).astype(np.int16)
    schema = Schema(
        n_param, n_branch,
        # def_prob=def_prob,
        # def_path=def_path,
        key_rep_type='time'
    )
    schema.key_rep_type
    key, val = schema.sample()
    print(key)
    print(val)
    print(schema.transition)
