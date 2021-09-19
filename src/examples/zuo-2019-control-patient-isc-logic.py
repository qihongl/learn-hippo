'''hypothetical control-patient isc in Zuo et al. 2020'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='poster', palette='colorblind')


def sample(n_time_steps, mu=0, sigma=1, x0=0):
    x = np.zeros(n_time_steps,)
    x[0] = x0
    for t in np.arange(1, n_time_steps):
        x[t] = x[t - 1] + np.random.normal(loc=mu, scale=sigma)
    return x


'''logic, compare control-patient isc'''
seed_val = 0
np.random.seed(seed_val)
n_time_steps = 200
noise_scale = 3

# gen data
x = sample(n_time_steps)
noise1 = np.random.normal(size=np.shape(x), scale=noise_scale)
noise2 = np.random.normal(size=np.shape(x), scale=noise_scale)

# plot control only
f, ax = plt.subplots(1, 1, figsize=(10, 3.5))
ax.plot(x + noise1)
# ax.plot(x + noise)
# ax.legend(['control', 'patient'])
ax.set_xlabel('Time')
ax.set_ylabel('BOLD')
ax.set_ylim([-10, 25])
ax.set_xticks([])
ax.set_yticks([])
sns.despine()
f.tight_layout()
f.savefig('examples/figs/isc-c.png', dpi=120)

# plot control and patient
f, ax = plt.subplots(1, 1, figsize=(10, 3.5))
ax.plot(x + noise1)
ax.plot(x + noise2)
# ax.legend(['control', 'patient'])
ax.set_xlabel('Time')
ax.set_ylabel('BOLD')
ax.set_ylim([-10, 25])
ax.set_xticks([])
ax.set_yticks([])
sns.despine()
f.tight_layout()
f.savefig('examples/figs/isc-cp.png', dpi=120)
# f.savefig('examples/figs/isc-2.png', dpi=120)


'''logic, time scrabling'''
seed_val = 15
np.random.seed(seed_val)
y = sample(n_time_steps)

f, ax = plt.subplots(1, 1, figsize=(10, 3.5))
ax.plot(x + noise1)
ax.plot(y + noise2)
ax.set_xticks([])
ax.set_yticks([])
sns.despine()
f.tight_layout()
ax.set_xlabel('Time')
ax.set_ylabel('BOLD')

'''2 subjects'''
np.random.seed(seed_val)
T = 50
n_units = 3
signal = 1
noise = .1
act_i = np.random.normal(loc=0, scale=signal, size=(T, n_units))
act_j = np.random.normal(loc=0, scale=signal, size=(T, n_units))
act_shared = np.random.normal(loc=0, scale=signal, size=(T, n_units))
s_i = act_shared + np.random.normal(loc=0, scale=noise, size=(T, n_units))
s_j = act_shared + np.random.normal(loc=0, scale=noise, size=(T, n_units))

f, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
axes[0].plot(act_i)
axes[1].plot(act_j)
axes[0].set_ylabel('Activity, net i')
axes[1].set_ylabel('Activity, net j')
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
ax.set_xlabel('Time')
sns.despine()
axes[0].set_title('Native space')
f.tight_layout()

f, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
axes[0].plot(s_i)
axes[1].plot(s_j)
axes[0].set_ylabel('Activity, net i')
axes[1].set_ylabel('Activity, net j')
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
ax.set_xlabel('Time')
axes[0].set_title('Shared space')
sns.despine()
f.tight_layout()
