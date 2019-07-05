import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from task import SequenceLearning
from analysis import compute_stats
from matplotlib.ticker import FormatStrFormatter
sns.set(style='white', palette='colorblind', context='poster')

'''study inter-event similarity as a function of n_branch, n_param'''

n_param = 15
n_branch = 4
pad_len = 0
# init
task = SequenceLearning(
    n_param=n_param, n_branch=n_branch, pad_len=pad_len, n_parts=1
)
# sample
n_samples = 200
X, Y = task.sample(n_samples, to_torch=False)
# unpack
print(f'X shape: {np.shape(X)}, n_samples x T x x_dim')
print(f'Y shape: {np.shape(Y)},  n_samples x T x y_dim')

'''take one sample, visualize it
note that
- if q keys are oredered
- then at (t+pad_len)-th time point, the query key is t

- so the uncertainty is low
- iff the 1st few o keys contain info about early time points
'''
i = 0
X_i = X[i]
np.shape(X_i)

o_key = X_i[:, :task.k_dim]
q_key = X_i[:, -task.k_dim:]

f, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
sns.heatmap(o_key, ax=axes[0])
sns.heatmap(q_key, ax=axes[1])
axes[0].set_title('o keys')
axes[1].set_title('q keys')
f.tight_layout()

'''compute vectorized representation'''


def compute_n_known(X_i):
    T_part, x_dim = np.shape(X_i)
    pad_len_i = T_part - task.n_param
    # get the observation / query keys
    o_key = X_i[:, :task.k_dim]
    q_key = X_i[:, -task.k_dim:]
    o_key_int = np.argmax(o_key, axis=1)
    q_key_int = np.argmax(q_key, axis=1)
    # o_key_int[-pad_len_i:] = np.nan
    # q_key_int[:pad_len_i] = np.nan
    # compute n_knowns
    o_key_int_t, q_key_int_t = set(), set()
    n_knowns = np.zeros((T_part,))
    cur_knows = np.full((n_param,), np.nan)
    # loop over time
    for t in np.arange(1, T_part+1):
        # don't update if pass n_params
        if t < n_param+1:
            o_key_int_t = set(o_key_int[:t])
        # don't update before padding/delay
        if t > pad_len_i:
            q_key_int_t = set(q_key_int[pad_len_i:t])
            cur_knows[t-pad_len_i-1] = 1 if q_key_int[t-1] in o_key_int_t else 0
        # compute observation-query overlap ==> "known"
        n_knowns[t-1] = len(o_key_int_t & q_key_int_t)
        # print(t)
        # print(o_key_int_t)
        # print(q_key_int_t)
        # print(n_knowns[t-1])
        # print()
    return n_knowns, cur_knows, pad_len_i


n_knowns_list = [None] * n_samples
n_knowns = np.zeros((n_samples, n_param))
cur_knowns = np.zeros((n_samples, n_param))
pad_lens = np.zeros(n_samples, dtype=np.int)
for i in range(n_samples):
    n_knowns_list[i], cur_knowns[i], pad_lens[i] = compute_n_known(X[i])
    n_knowns[i, :] = n_knowns_list[i][pad_lens[i]:]

# plot # don't knows
n_dks = (n_param-n_knowns)/n_param
n_dks_mu, n_dks_er = compute_stats(n_dks, n_se=3)
b_pal = sns.color_palette('Blues', n_colors=5)
f, axes = plt.subplots(2, 1, figsize=(7, 10))
axes[0].plot(n_dks.T, alpha=.1, color=b_pal[1])
axes[0].errorbar(
    x=np.arange(n_param), y=n_dks_mu, yerr=n_dks_er,
    color=b_pal[-1]
)
title = f'Overall uncertainty\ndelay = {pad_len}, graph len = {n_param}'
axes[0].set_title(title)
axes[0].set_ylabel('% param unknow')

# plot P(don't knows | t)
cur_dk = 1-cur_knowns
p_dks_mu, p_dks_er = compute_stats(cur_dk, n_se=3)
axes[1].errorbar(
    x=np.arange(n_param), y=p_dks_mu, yerr=p_dks_er,
    color=b_pal[-1]
)
title = f'Current uncertainty\ndelay = {pad_len}, graph len = {n_param}'
axes[1].set_title(title)
axes[1].set_xlabel('Time, recall phase')
axes[1].set_ylabel('P(don\'t know current query)')

sns.despine()
f.tight_layout()


'''uncertainty curve as a function of delay'''

n_samples = 300
pad_len_list = [0, 2, 4, 8]

f, axes = plt.subplots(2, 1, figsize=(7, 10))

for pad_len in pad_len_list:
    # init
    task = SequenceLearning(
        n_param=n_param, n_branch=n_branch, pad_len=pad_len, n_parts=1
    )
    X, Y = task.sample(n_samples, to_torch=False)

    #
    n_knowns_list = [None] * n_samples
    n_knowns = np.zeros((n_samples, n_param))
    cur_knowns = np.zeros((n_samples, n_param))
    pad_lens = np.zeros(n_samples, dtype=np.int)
    for i in range(n_samples):
        n_knowns_list[i], cur_knowns[i], pad_lens[i] = compute_n_known(X[i])
        n_knowns[i, :] = n_knowns_list[i][pad_lens[i]:]

    n_dks = (n_param-n_knowns) / n_param
    n_dks_mu, n_dks_er = compute_stats(n_dks, n_se=3)
    axes[0].errorbar(x=range(n_param), y=n_dks_mu, yerr=n_dks_er)

    cur_dk = 1-cur_knowns
    p_dks_mu, p_dks_er = compute_stats(cur_dk, n_se=3)
    axes[1].errorbar(x=range(n_param), y=p_dks_mu, yerr=p_dks_er)

axes[1].legend(pad_len_list, title='delay', fancybox=True)
axes[1].set_xlabel('Time, recall phase')
axes[0].set_ylabel('% param unknow')
axes[1].set_ylabel('P(don\'t know current query)')
axes[0].set_title(f'Overall uncertainty, graph len = {n_param}')
axes[1].set_title(f'Current uncertainty, graph len = {n_param}')
sns.despine()
f.tight_layout()


# f, ax = plt.subplots(1, 1, figsize=(7, 5))
# for pad_len in pad_len_list:
#     # init
#     task = SequenceLearning(
#         n_param=n_param, n_branch=n_branch, pad_len=pad_len, n_parts=1
#     )
#     X, Y = task.sample(n_samples, to_torch=False)
#
#     #
#     n_knowns_list = [None] * n_samples
#     n_knowns = np.zeros((n_samples, n_param))
#     pad_lens = np.zeros(n_samples, dtype=np.int)
#     for i in range(n_samples):
#         n_knowns_list[i], cur_knowns[i], pad_lens[i] = compute_n_known(X[i])
#         n_knowns[i, :] = n_knowns_list[i][pad_lens[i]:]
#
#     n_dks = (n_param-n_knowns) / n_param
#     n_dks_mu, n_dks_er = compute_stats(n_dks, n_se=3)
#     ax.plot(n_dks_er)
#
# ax.legend(pad_len_list, title='delay', fancybox=True)
# ax.set_xlabel('Time, recall phase')
# ax.set_ylabel('variance')
# title = f'Uncertainty, graph len = {n_param}'
# ax.set_title(title)
# sns.despine()
# f.tight_layout()
