'''simulation: softmax output nan'''
import torch
import numpy as np
import torch.nn.functional as F
from models.A2C import _softmax, _pick_action

n_actions = 3
beta = 1

pi_a = torch.tensor([1e+9999999, 1e-9999]).type(torch.FloatTensor)
# pi_a = torch.tensor([np.inf, np.inf])
# pi_a = torch.randn(n_actions,)

softmax_pi_a = F.softmax(torch.squeeze(pi_a / beta), dim=0)
softmax_pi_a = F.log_softmax(torch.squeeze(pi_a / beta), dim=0)

# if torch.any(torch.isnan(softmax_pi_a)):
#     raise ValueError(f'Softmax causing nan: {pi_a} -> {softmax_pi_a}')

print('before :', pi_a)
print('after  :', softmax_pi_a)

# a, log_p = _pick_action(softmax_pi_a)
# print('action: ', a)
# print('log p : ', log_p)
