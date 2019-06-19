# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task.StimSampler import StimSampler
sns.set(style='white', palette='colorblind', context='talk')

'''test'''

# init a graph
n_param, n_branch = 3, 2
n_timesteps = n_param
n_parts = 2
p_rm_ob_enc, p_rm_ob_rcl = .25, 0

sampler = StimSampler(n_param, n_branch)
sample_ = sampler.sample(
    n_parts,
    p_rm_ob_enc, p_rm_ob_rcl,
)
[o_keys_vec, o_vals_vec], [q_keys_vec, q_vals_vec] = sample_

# plot
cmap = 'bone'
rk, rv = n_param * n_branch, n_branch
f, axes = plt.subplots(
    n_parts, 4, figsize=(10, 4), sharey=True,
    gridspec_kw={'width_ratios': [rk, rv, rk, rv]}
)
for ip in range(n_parts):
    axes[ip, 0].imshow(o_keys_vec[ip], cmap=cmap)
    axes[ip, 1].imshow(o_vals_vec[ip], cmap=cmap)
for ip in range(n_parts):
    axes[ip, 2].imshow(q_keys_vec[ip], cmap=cmap)
    axes[ip, 3].imshow(q_vals_vec[ip], cmap=cmap)
# label
# axes[0, 0].set_title('Observation')
# axes[0, 2].set_title('Queries')
axes[-1, 0].set_xlabel('Keys/States')
axes[-1, 1].set_xlabel('Values/Action')
axes[-1, 2].set_xlabel('Keys/States')
axes[-1, 3].set_xlabel('Values/Action')
# modify y ticks/labels
for ip in range(n_parts):
    axes[ip, 0].set_yticks(range(n_timesteps))
    axes[ip, 0].set_yticklabels(range(n_timesteps))
    axes[ip, 0].set_ylabel(f'Time, part {ip+1}')
f.subplots_adjust(wspace=.1, hspace=.4)
f.savefig('examples/figs/movie-sample-human.png', dpi=100, bbox_inches='tight')
