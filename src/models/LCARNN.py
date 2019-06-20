"""
LCA LSTM
"""
import torch
import torch.nn as nn

from torch.distributions import Categorical
from models.DND import DND
from models.A2C import A2C, A2C_linear
from models.initializer import ortho_init, xavier_uniform_init, set_forget_gate_bias

# constants
# number of vector signal (lstm gates)
N_VSIG = 3
# number of scalar signal (sigma)
N_SSIG = 3
N_LCA_SIG = 3
# the ordering in the cache
scalar_signal_names = ['input strength', 'leak', 'competition']
vector_signal_names = ['f', 'i', 'o']


class LCARNN(nn.Module):

    def __init__(
            self,
            input_dim, n_hidden, n_action,
            recall_func='LCA',
            kernel='cosine',
            dict_len=100,
            weight_init_scheme='ortho',
            init_state_trainable=False,
            layernorm=False,
            mode='train',
            a2c_linear=True,
            bias=True
    ):
        super(LCARNN, self).__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.bias = bias
        self.rnn = nn.LSTM(input_dim, n_hidden)
        # hpc
        self.hpc_ctrl = nn.Linear(n_hidden, N_LCA_SIG)
        # memory
        self.dnd = DND(dict_len, n_hidden, kernel, recall_func)
        # the RL mechanism
        if a2c_linear:
            self.a2c = A2C_linear(n_hidden, n_action)
        else:
            self.a2c = A2C(n_hidden, n_hidden, n_action)
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
        set_forget_gate_bias(self.rnn)

    def init_init_states(self):
        scale = 1 / self.n_hidden
        self.h_0 = torch.nn.Parameter(
            torch.randn(1, 1, self.n_hidden)*scale, requires_grad=True
        )
        self.c_0 = torch.nn.Parameter(
            torch.randn(1, 1, self.n_hidden)*scale, requires_grad=True
        )

    def get_init_states(self, scale=.1, device='cpu'):
        if self.init_state_trainable:
            return self.h_0, self.c_0
        else:
            h_0 = torch.randn(1, 1, self.n_hidden).to(device) * scale
            c_0 = torch.randn(1, 1, self.n_hidden).to(device) * scale
            return h_0, c_0

    def forward(self, x_t, hc, beta=1):
        # x_t = x_t.view(1, 1, -1)
        h_t, hc_t = self.rnn(x_t, hc)
        # memory actions
        theta = self.hpc_ctrl(h_t).sigmoid()
        [inps_t, leak_t, comp_t] = torch.squeeze(theta)
        # recall / encode
        h_t, m_t = self.recall(h_t, leak_t, comp_t, inps_t)
        self.encode(h_t)
        # policy
        action_dist_t, value_t = self.a2c.forward(h_t, beta=beta)
        # scache results
        scalar_signal = [inps_t, leak_t, comp_t]
        misc = [m_t]
        vector_signal = []
        cache = [vector_signal, scalar_signal, misc]
        # update
        hc_t = (h_t, hc_t[1])
        return action_dist_t, value_t, hc_t, cache

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
            lure_memory = torch.randn(1, 1, self.n_hidden)
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
