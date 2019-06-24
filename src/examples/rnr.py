import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from task.RNR import RNR
from utils.constants import RNR_COND_DICT
sns.set(style='white', palette='colorblind', context='talk')

'''testing'''

n_param, n_branch = 6, 3
n_parts = 3
p_rm_ob_enc = 0
p_rm_ob_rcl = 0
n_samples = 5
# context_dim = 10
append_context = True
task = RNR(
    n_param, n_branch,
    context_onehot=False,
    context_drift=True,
    context_dim=5,
    append_context=append_context,
)

# take sample
stack = False
data_batch_ = task._make_rnr_batch(stack=stack)
x_batch, y_batch, rcl_mv_id_batch, cond_id_batch = data_batch_
np.shape(x_batch)
i = 0
x_i, y_i = x_batch[i], y_batch[i]
rcl_mv_id_i, cond_id_i = rcl_mv_id_batch[i], cond_id_batch[i]


'''make a plot'''

cmap = 'bone'
if stack:
    _, x_dim = np.shape(x_i)
    f, axes = plt.subplots(
        1, 2, figsize=(7, 7), sharey=True,
        gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]})
    axes[0].imshow(x_i, cmap=cmap, vmin=0, vmax=1)
    axes[1].imshow(y_i, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_ylabel('Time')

    ox_label = 'key | val | ctx' if append_context else 'key | val'
    axes[0].set_xlabel(ox_label)
    axes[1].set_xlabel('val')
    axes[0].axvline(task.k_dim-.5, color='red', linestyle='--')
    axes[0].axvline(task.k_dim+task.v_dim-.5, color='red', linestyle='--')
    for ax in axes:
        for ip in range(task.n_parts-1):
            ax.axhline(task.T_part * (ip+1) - .5, color='red', linestyle='--')

else:
    n_parts_, n_timesteps_, x_dim = np.shape(x_i)
    f, axes = plt.subplots(
        3, 2, figsize=(7, 9), sharey=True,
        gridspec_kw={'width_ratios': [task.x_dim, task.y_dim]})

    for ip in range(n_parts):
        axes[ip, 0].imshow(x_i[ip], cmap=cmap, vmin=0, vmax=1)
        axes[ip, 1].imshow(y_i[ip], cmap=cmap, vmin=0, vmax=1)
        axes[ip, 0].set_ylabel(f'time, part {ip}')
        axes[ip, 0].axvline(task.k_dim-.5, color='red', linestyle='--')
        axes[ip, 0].axvline(task.k_dim+task.v_dim-.5,
                            color='red', linestyle='--')

    ox_label = 'key | val | ctx' if append_context else 'key | val'
    axes[-1, 0].set_xlabel(ox_label)
    axes[-1, 1].set_xlabel('val')

f.suptitle(
    f'cond = {RNR_COND_DICT[cond_id_i]}, memory id = {rcl_mv_id_i}',
    fontsize=16
)
f.tight_layout()
