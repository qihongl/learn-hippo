""" A leaky competing accumulator.

Notes:
1. This is a pytorch implementation, so it is autodiff-able.
- This requires avoiding any in-place update as well.

2. For my own project, I made this version to be a noiseless process.

References:
[1] Usher, M., & McClelland, J. L. (2001).
The time course of perceptual choice: the leaky, competing accumulator model.
Psychological Review, 108(3), 550–592.
[2] Polyn, S. M., Norman, K. A., & Kahana, M. J. (2009).
A context maintenance and retrieval model of organizational processes in free
recall. Psychological Review, 116(1), 129–156.
[3] PsyNeuLink: https://github.com/PrincetonUniversity/PsyNeuLink
"""
import torch


class LCA():
    """The leaky competing accumulator class in pytorch.
    """

    def __init__(
        self,
        n_units, leak, ltrl_inhib,
        self_excit=0, w_input=1, w_cross=0, offset=0,
        dt_t=.6, noise_sd=0, threshold=1
    ):
        """Initialize a leaky competing accumulator.

        Parameters
        ----------
        n_units : int
            the number of accumulators in the LCA
        dt_t : float
            the dt / tao term, representing the time step size
        leak : float
            the leak term
        ltrl_inhib : float
            the lateral inhibition across accumulators (i vs. j)
        self_excit : float
            the self excitation of a accumulator (i vs. i)
        w_input : float
            input strengh of the feedforward weights
        w_cross : float
            cross talk of the feedforward weights
        offset : float
            the additive drift term of the LCA process
        noise_sd : float
            the sd of the noise term of the LCA process

        """
        self.n_units = n_units
        self.dt_t = dt_t
        self.leak = leak
        self.ltrl_inhib = ltrl_inhib
        self.self_excit = self_excit
        self.w_input = w_input
        self.w_cross = w_cross
        self.offset = offset
        self.noise_sd = noise_sd
        self.threshold = threshold
        # the input / recurrent weights
        self.W_i = make_weights(w_input, w_cross, n_units)
        self.W_r = make_weights(self_excit, -ltrl_inhib, n_units)
        # check params
        self._check_model_config()

    def run(self, stimuli):
        """Run LCA on some stimulus sequence
        the update formula:
            1. value =   prev value
                      + new_input
                      - leaked previous value
                      + previous value updated with recurrent weigths
                      + offset and noise
            2. value <- output_bounding(value)

        Parameters
        ----------
        stimuli : 2d array
            input sequence, with shape: T x n_units
        threshold: float
            the upper bound of the neural activity

        Returns
        -------
        2d array
            LCA acitivity time course, with shape = np.shape(stimuli)

        """
        # input validation
        self._check_inputs(stimuli)
        T, _ = stimuli.size()
        # # precompute sqrt(dt/tao)
        # sqrt_dt_t = self.dt_t ** .5
        # # precompute noise for all time points
        # noise = torch.randn(size=(T, self.n_units)) * self.noise_sd * sqrt_dt_t
        # precompute the transformed input, for all time points
        inp = torch.matmul(stimuli, self.W_i)
        # precompute offset for all units
        offset = self.offset * torch.ones(self.n_units,)
        # loop over n_cycles
        init_val = torch.zeros(self.n_units,)
        # prealloc values for the accumulators over time
        V = torch.zeros((T, self.n_units))
        for t in range(T):
            # the LCA computation at time t
            V_prev = init_val if t == 0 else V_bd
            V_cur = V_prev + offset + \
                (inp[t, :] - self.leak * V_prev +
                 torch.matmul(self.W_r, V_prev)) * self.dt_t
            # V_cur += noise[t, :]
            # output bounding
            # V_bd = V_cur.sigmoid()
            V_relu = torch.max(V_cur, torch.zeros(self.n_units,))
            V_bd = torch.min(V_relu, torch.ones(self.n_units,) * self.threshold)
            V[t, :] = V_bd
        return V

    def _check_model_config(self):
        assert 0 <= self.leak, f'Invalid leak = {self.leak}'
        assert 0 <= self.ltrl_inhib, f'Invalid ltrl_inhib = {self.ltrl_inhib}'
        assert 0 < self.dt_t, f'Invalid dt_t = {self.dt_t}'
        assert 0 <= self.noise_sd, f'Invalid noise sd = {self.noise_sd}'
        assert 0 < self.threshold, f'Invalid threshold = {self.threshold}'

    def _check_inputs(self, stimuli):
        assert torch.is_tensor(stimuli), \
            'stimuli has to be a pytorch tensor'
        _, n_units_ = stimuli.size()
        assert n_units_ == self.n_units,\
            f'stimuli shape inconsistent with the network size = {self.leak}'

    def __repr__(self):
        return f"""LCA
        n_units = {self.n_units}
        dt_t ={self.dt_t}
        leak = {self.leak}
        ltrl_inhib = {self.ltrl_inhib}
        self_excit = {self.self_excit}
        w_input = {self.w_input}
        w_cross = {self.w_cross}
        offset = {self.offset}
        noise_sd = {self.noise_sd}
        """


def make_weights(diag_val, offdiag_val, n_units):
    """Get a connection weight matrix with "diag-offdial structure"

    e.g.
        | x, y, y |
        | y, x, y |
        | y, y, x |

    where x = diag_val, and y = offdiag_val

    Parameters
    ----------
    diag_val : float
        the value of the diag entries
    offdiag_val : float
        the value of the off-diag entries
    n_units : int
        the number of LCA nodes

    Returns
    -------
    2d array
        the weight matrix with "diag-offdial structure"

    """
    diag_mask = torch.eye(n_units)
    offdiag_mask = torch.ones((n_units, n_units)) - torch.eye(n_units)
    weight_matrix = diag_mask * diag_val + offdiag_mask * offdiag_val
    return weight_matrix.float()
