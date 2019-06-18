import numpy as np


class ReplayBuffer:
    """basically a fancy list

    Parameters
    ----------
    size : int
        max number of experience

    """

    def __init__(self, size=1000):
        self.buffer = []
        self.size = size

    def flush_buffer(self):
        self.buffer = []

    def append(self, experience):
        """add an expereince to the replay buffer.

        Parameters
        ----------
        experience : list
            e.g. for non-recurrent Q network, might add ...
                [s_t, a_t, r_t, s_next, terminate]

        """
        self.buffer.append(experience)
        # handle overflow
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def sample(self, mode='uniform'):
        """sample a previously stored experience.
        TODO implement prioritized sampling

        Parameters
        ----------
        mode : string
            the sampling mode, default to uniform sampling

        Returns
        -------
        list
            an experience / transition

        """
        if mode == 'uniform':
            sample_id = np.random.randint(len(self.buffer))
        else:
            raise ValueError('unrecognized sampling mode')
        return self.buffer[sample_id]
