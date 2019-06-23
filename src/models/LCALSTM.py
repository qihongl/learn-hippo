"""
LCA LSTM
"""
import torch
import torch.nn as nn

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


class LCALSTM(nn.Module):

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
        super(LCALSTM, self).__init__()
        self.weight_init_scheme = weight_init_scheme
        self.init_state_trainable = init_state_trainable
        # dims
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.bias = bias
        # rnn
        self.lstm = nn.LSTM(input_dim, n_hidden)
        # hpc control layer and episodic memory
        self.hpc_ctrl = nn.Linear(n_hidden, N_LCA_SIG)
        self.dnd = DND(dict_len, n_hidden, kernel, recall_func)
        # the RL mechanism
        if a2c_linear:
            self.a2c = A2C_linear(n_hidden, n_action)
        else:
            self.a2c = A2C(n_hidden, n_hidden, n_action)
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
        set_forget_gate_bias(self.lstm)

    def init_init_states(self):
        scale = 1 / self.n_hidden
        self.h_0 = torch.nn.Parameter(
            sample_random_vector(self.n_hidden, scale), requires_grad=True
        )
        self.c_0 = torch.nn.Parameter(
            sample_random_vector(self.n_hidden, scale), requires_grad=True
        )

    def get_init_states(self, scale=.1, device='cpu'):
        if self.init_state_trainable:
            h_0_, c_0_ = self.h_0, self.c_0
        else:
            h_0_ = sample_random_vector(self.n_hidden, scale)
            c_0_ = sample_random_vector(self.n_hidden, scale)
        return (h_0_, c_0_)

    def forward(self, x_t, hidden_state_prev, beta=1):
        # recurrent
        rnn_out_t, hidden_state_t = self.lstm(x_t, hidden_state_prev)
        # memory actions
        theta = self.hpc_ctrl(rnn_out_t).sigmoid()
        [inps_t, leak_t, comp_t] = torch.squeeze(theta)
        # recall / encode
        mem_t = self.recall(rnn_out_t, leak_t, comp_t, inps_t)
        des_act_t = rnn_out_t + mem_t
        self.encode(des_act_t)
        # policy
        action_dist_t, value_t = self.a2c.forward(des_act_t, beta=beta)
        # scache results
        vector_signal = []  # [gates]
        scalar_signal = [inps_t, leak_t, comp_t]
        misc = [rnn_out_t, mem_t, des_act_t]
        cache = [vector_signal, scalar_signal, misc]
        # update
        hidden_state_t = (rnn_out_t, hidden_state_t[1])
        return action_dist_t, value_t, hidden_state_t, cache

    def pick_action(self, action_distribution):
        a_t, log_prob_a_t = self.a2c.pick_action(action_distribution)
        return a_t, log_prob_a_t

    def recall(self, input_pattern, leak_t, comp_t, inps_t):
        """run the "pattern completion" procedure

        Parameters
        ----------
        input_pattern : torch.tensor, vector
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
            memory = torch.zeros_like(input_pattern).data
        else:
            memory = self.dnd.get_memory(
                input_pattern,
                leak=leak_t, comp=comp_t, w_input=inps_t
            )
        return memory

    def encode(self, input_pattern):
        if not self.dnd.encoding_off:
            self.dnd.save_memory(input_pattern, input_pattern)

    def init_em_config(self):
        self.flush_episodic_memory()
        self.encoding_off()
        self.retrieval_off()

    def add_simple_lures(self, n_lures=1):
        for _ in range(n_lures):
            lure_i = sample_random_vector(self.n_hidden)
            self.dnd.inject_memories([lure_i], [lure_i])

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

    # def __repr__(self):
    #     return s


def sample_random_vector(n_dim, scale=.1):
    return torch.randn(1, 1, n_dim) * scale
