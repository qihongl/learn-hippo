import torch
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from utils.utils import to_np
from models import get_reward
from task import SequenceLearning
# sns.set(style='white', palette='colorblind', context='talk')
# np.random.seed(2)

'''how to use'''
# init
n_param, n_branch = 5, 3
pad_len = 3
n_parts = 2
n_samples = 5
penalty = 4

# take sample
task = SequenceLearning(n_param, n_branch, pad_len=pad_len)
X, Y = task.sample(n_samples, to_torch=True)

# pick a sample
i = 0
t = 4

y_it = Y[i][t]
print('target: ', y_it)
for a_t_np in range(n_branch+1):
    a_t = torch.tensor(a_t_np)
    r_t = get_reward(a_t, y_it, penalty)
    print('a_t/r_t: ', a_t, r_t)


# Y[i] = to_np(Y[i])
# to_np(Y)
