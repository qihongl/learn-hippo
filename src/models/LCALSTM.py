"""
LCA LSTM
"""
import torch
import torch.nn as nn

from models.DND import DND
from models.A2C import A2C
from torch.distributions import Categorical
from models.initializer import initialize_weights

# constants
# number of vector signal (lstm gates)
N_VSIG = 3
# number of scalar signal (sigma)
N_SSIG = 3
# the ordering in the cache
scalar_signal_names = ['input strength', 'leak', 'competition']
vector_signal_names = ['f', 'i', 'o']
#
sigmoid = nn.Sigmoid()
gain = 1


class LCALSTM(nn.Module):

    def __init__(
            self,
            input_dim, hidden_dim, output_dim,
            recall_func='LCA', kernel='cosine', dict_len=100,
            weight_init_scheme='ortho',
            init_state_trainable=False,
            layernorm=False,
            a2c_linear=False,
            bias=True
    ):
        super(LCALSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.n_units_total = (N_VSIG+1) * hidden_dim + N_SSIG
        # input-hidden weights
        self.i2h = nn.Linear(input_dim, self.n_units_total, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(hidden_dim, self.n_units_total, bias=bias)
        # normalization
        self.layernorm = layernorm
        if layernorm:
            self.ln = nn.LayerNorm((1, hidden_dim))
        # memory
        self.dnd = DND(dict_len, hidden_dim, kernel, recall_func)
        # the RL mechanism
        self.a2c = A2C(hidden_dim, hidden_dim, output_dim)
        self.weight_init_scheme = weight_init_scheme
        self.init_state_trainable = init_state_trainable
        self.init_model()

    def init_model(self):
        # add name fields
        self.n_ssig = N_SSIG
        self.n_vsig = N_VSIG
        self.vsig_names = vector_signal_names
        self.ssig_names = scalar_signal_names
        # init params
        initialize_weights(self, self.weight_init_scheme)
        if self.init_state_trainable:
            self.init_init_states()

    def init_init_states(self):
        scale = 1 / self.hidden_dim
        self.h_0 = torch.nn.Parameter(
            sample_random_vector(self.hidden_dim, scale), requires_grad=True
        )
        self.c_0 = torch.nn.Parameter(
            sample_random_vector(self.hidden_dim, scale), requires_grad=True
        )

    def get_init_states(self, scale=.1, device='cpu'):
        if self.init_state_trainable:
            h_0_, c_0_ = self.h_0, self.c_0
        else:
            h_0_ = sample_random_vector(self.hidden_dim, scale)
            c_0_ = sample_random_vector(self.hidden_dim, scale)
        return (h_0_, c_0_)

    def forward(self, x_t, hc_prev, beta=1):
        # unpack activity
        (h_prev, c_prev) = hc_prev
        h_prev = h_prev.view(h_prev.size(1), -1)
        c_prev = c_prev.view(c_prev.size(1), -1)
        x_t = x_t.view(x_t.size(1), -1)
        # transform the input info
        preact = self.i2h(x_t) + self.h2h(h_prev)
        # get all gate values
        gates = preact[:, : N_VSIG * self.hidden_dim].sigmoid()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.hidden_dim]
        o_t = gates[:, self.hidden_dim:2 * self.hidden_dim]
        i_t = gates[:, -self.hidden_dim:]
        # get kernel param
        inps_t = sigmoid(preact[:, N_VSIG * self.hidden_dim+0] * gain)
        leak_t = sigmoid(preact[:, N_VSIG * self.hidden_dim+1] * gain)
        comp_t = sigmoid(preact[:, N_VSIG * self.hidden_dim+2] * gain)
        # stuff to be written to cell state
        c_t_new = preact[:, N_VSIG * self.hidden_dim+N_SSIG:].tanh()
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(c_prev, f_t) + torch.mul(i_t, c_t_new)
        # recall
        m_t = self.recall(c_t, leak_t, comp_t, inps_t)
        cm_t = c_t + m_t
        # encode
        self.encode(cm_t)
        # normalize activity
        if self.layernorm:
            cm_t = self.ln(cm_t)
        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, cm_t.tanh())
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        cm_t = cm_t.view(1, cm_t.size(0), -1)
        # produce action distribution and value estimate
        a2c_outputs_ = self.a2c.forward(h_t, beta=beta, return_h=True)
        [action_dist_t, value_t, decision_activity_t] = a2c_outputs_
        # scache results
        scalar_signal = [inps_t, leak_t, comp_t]
        vector_signal = [f_t, i_t, o_t]
        misc = [h_t, m_t, cm_t, decision_activity_t, self.dnd.vals]
        cache = [vector_signal, scalar_signal, misc]
        return action_dist_t, value_t, (h_t, cm_t), cache

    def recall(self, c_t, leak_t, comp_t, inps_t):
        """run the "pattern completion" procedure

        Parameters
        ----------
        c_t : torch.tensor, vector
            cell state
        leak_t : torch.tensor, scalar
            LCA param, leak
        comp_t : torch.tensor, scalar
            LCA param, lateral inhibition
        inps_t : torch.tensor, scalar
            LCA param, input strength / feedforward weights

        Returns
        -------
        tensor, tensor
            updated cell state, recalled item

        """
        if self.dnd.retrieval_off:
            m_t = torch.zeros_like(c_t)
        else:
            # retrieve memory
            m_t = self.dnd.get_memory(
                c_t, leak=leak_t, comp=comp_t, w_input=inps_t
            )
        return m_t

    def encode(self, cm_t):
        if not self.dnd.encoding_off:
            self.dnd.save_memory(cm_t, cm_t)

    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.

        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)

        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)

        """
        m = Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t

    def inject_memories(self, keys, vals):
        self.dnd.inject_memories(keys, vals)

    def add_simple_lures(self, n_lures=1):
        for _ in range(n_lures):
            lure_i = sample_random_vector(self.hidden_dim)
            self.dnd.inject_memories([lure_i], [lure_i])

    def init_em_config(self):
        self.flush_episodic_memory()
        self.encoding_off()
        self.retrieval_off()

    def flush_episodic_memory(self):
        self.dnd.flush()

    def encoding_off(self):
        self.dnd.encoding_off = True

    def retrieval_off(self):
        self.dnd.retrieval_off = True

    def encoding_on(self):
        self.dnd.encoding_off = False

    def retrieval_on(self):
        self.dnd.retrieval_off = False


def sample_random_vector(n_dim, scale=.1):
    return torch.randn(1, 1, n_dim) * scale
