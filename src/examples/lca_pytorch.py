import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from utils.utils import to_sqnp
from models.LCA_pytorch import LCA

sns.set(style='white', palette='colorblind', context='talk')
np.random.seed(0)


"""model params
"""
n_units = 3
# input weights
w_input = 1
# decision param
leak = .1
competition = 1
self_excit = 0
# time step size
dt = .4
#
self_excit = 0
w_cross = 0
offset = 0
noise_sd = 0


"""set up the input
"""
T = 10
input_pattern_set = list(np.eye(n_units))
stimuli = np.vstack([
    np.tile(input_pattern, (T, 1)) for input_pattern in input_pattern_set
])
stimuli = torch.tensor(stimuli).float()

# init LCA
lca = LCA(
    n_units, dt, leak, competition,
    self_excit=self_excit, w_input=w_input, w_cross=w_cross,
    offset=offset, noise_sd=noise_sd,
)
# run LCA
vals = lca.run(stimuli)

'''plot'''
# calc event boundaries
bounds = [(k+1)*T for k in np.arange(n_units-1)]
# plot
f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(to_sqnp(vals))
for b in bounds:
    ax.axvline(b, linestyle='--', color='grey')
title_text = f"""
The temporal dynamics of a {n_units}-unit LCA
each unit is turned on sequentially
"""
ax.set_title(title_text)
ax.set_xlabel('Time')
ax.set_ylabel('Activity')
# ax.set_ylim([-.05, 1.05])
f.tight_layout()
sns.despine()
