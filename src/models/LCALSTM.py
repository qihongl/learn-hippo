"""
LCA LSTM
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models.DND import DND
from models.A2C import A2C, A2C_linear
from models.initializer import ortho_init, xavier_uniform_init

# constants
# number of vector signal (lstm gates)
N_VSIG = 3
# number of scalar signal (sigma)
N_SSIG = 3
# the ordering in the cache
scalar_signal_names = ['input strength', 'leak', 'competition']
vector_signal_names = ['f', 'i', 'o']


class LCALSTM(nn.Module):

    def __init__(
            self,
            input_dim, hidden_dim, output_dim,
            recall_func, kernel, dict_len=100,
            weight_init_scheme='ortho',
            init_state_trainable=False,
            layernorm=False,
            mode='train',
            a2c_linear=True, predict=False,
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
        if a2c_linear:
            self.a2c = A2C_linear(hidden_dim, output_dim)
        else:
            self.a2c = A2C(hidden_dim, hidden_dim, output_dim)
        #
        self.predict = predict
        if predict:
            self.predictor = nn.Linear(hidden_dim, output_dim-1, bias=bias)
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
        self.init_weights(option=self.weight_init_scheme)
        if self.init_state_trainable:
            self.init_init_states()

    def init_weights(self, option='ortho'):
        if option == 'ortho':
            ortho_init(self)
        elif option == 'xaiver_uniform':
            xavier_uniform_init(self)
        else:
            raise ValueError(f'unrecognizable weight init scheme {option}')

    def init_init_states(self):
        scale = 1 / self.hidden_dim
        self.h_0 = torch.nn.Parameter(
            torch.randn(1, 1, self.hidden_dim)*scale, requires_grad=True
        )
        self.c_0 = torch.nn.Parameter(
            torch.randn(1, 1, self.hidden_dim)*scale, requires_grad=True
        )

    def get_init_states(self, scale=.1, device='cpu'):
        if self.init_state_trainable:
            return self.h_0, self.c_0
        else:
            h_0 = torch.randn(1, 1, self.hidden_dim).to(device) * scale
            c_0 = torch.randn(1, 1, self.hidden_dim).to(device) * scale
            return h_0, c_0

    def forward(self, x_t, hc, beta=1):
        # unpack activity
        (h, c) = hc
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x_t = x_t.view(x_t.size(1), -1)
        # transform the input info
        preact = self.i2h(x_t) + self.h2h(h)
        # get all gate values
        gates = preact[:, : N_VSIG * self.hidden_dim].sigmoid()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.hidden_dim]
        o_t = gates[:, self.hidden_dim:2 * self.hidden_dim]
        i_t = gates[:, -self.hidden_dim:]
        # get kernel param
        inps_t = preact[:, N_VSIG * self.hidden_dim+0].sigmoid()
        leak_t = preact[:, N_VSIG * self.hidden_dim+1].sigmoid()
        comp_t = preact[:, N_VSIG * self.hidden_dim+2].sigmoid()
        # stuff to be written to cell state
        c_t_new = preact[:, N_VSIG * self.hidden_dim+N_SSIG:].tanh()
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(c, f_t) + torch.mul(i_t, c_t_new)
        # recall
        c_t, m_t = self.recall(c_t, leak_t, comp_t, inps_t)
        # encode
        self.encode(c_t)
        # normalize activity
        if self.layernorm:
            c_t = self.ln(c_t)
        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, c_t.tanh())
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        # produce action distribution and value estimate
        action_dist_t, value_t = self.a2c.forward(h_t, beta=beta)
        # scache results
        scalar_signal = [inps_t, leak_t, comp_t]
        vector_signal = [f_t, i_t, o_t]
        misc = [m_t]
        cache = [vector_signal, scalar_signal, misc]
        if self.predict:
            yhat_t = self.predictor(h_t).sigmoid()
            return action_dist_t, value_t, yhat_t, (h_t, c_t), cache
        return action_dist_t, value_t, (h_t, c_t), cache

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
            cm_t = c_t
            m_t = torch.zeros_like(c_t)
        else:
            # retrieve memory
            m_t = self.dnd.get_memory(
                c_t, leak=leak_t, comp=comp_t, w_input=inps_t
            )
            cm_t = c_t + m_t
        return cm_t, m_t

    def encode(self, cm_t):
        if not self.dnd.encoding_off:
            self.dnd.save_memory(cm_t, cm_t)

    def init_em_config(self):
        self.flush_episodic_memory()
        self.encoding_off()
        self.retrieval_off()

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
        if n_lures <= 0:
            return
        for n in range(n_lures):
            lure_memory = torch.randn(1, 1, self.hidden_dim)
            self.inject_memories([lure_memory], [lure_memory])

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

    def to_train_mode(self):
        self.train()

    def to_test_mode(self):
        self.eval()
