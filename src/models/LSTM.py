"""
Implement a lstm from scratch for customization purpose
- the initial code is adapted from pytorch source code
"""
import numpy as np
import torch as torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, bias=True):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        # input-hidden weights
        self.i2h = nn.Linear(input_dim, 4 * hidden_dim, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        # init
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() > 1:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x_t, h, c):
        # unpack activity
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x_t = x_t.view(x_t.size(1), -1)
        # Linear mappings
        preact = self.i2h(x_t) + self.h2h(h)
        # get all gate values
        gates = preact[:, :3 * self.hidden_dim].sigmoid()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.hidden_dim]
        i_t = gates[:, self.hidden_dim:2 * self.hidden_dim]
        o_t = gates[:, -self.hidden_dim:]
        # stuff to be written to cell state
        c_t_new = preact[:, 3 * self.hidden_dim:].tanh()
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(c, f_t) + torch.mul(i_t, c_t_new)
        # gated hidden state
        h_t = torch.mul(o_t, c_t.tanh())
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        # output, in general, can be any diff-able transformation of h_t
        output_t = h_t
        # fetch activity
        cache = [f_t, i_t, o_t]
        return output_t, h_t, c_t, cache
