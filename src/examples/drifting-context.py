import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from task.Schema import sample_context_drift
sns.set(style='white', palette='colorblind', context='talk')


def colored_line(x, y, z=None, linewidth=1, MAP='Blues'):
    '''refereince: https://stackoverflow.com/a/27537018'''
    # this uses pcolormesh to make interpolated rectangles
    xl = len(x)
    [xs, ys, zs] = [np.zeros((xl, 2)), np.zeros((xl, 2)), np.zeros((xl, 2))]

    # z is the line length drawn or a list of vals to be plotted
    if z is None:
        z = [0]

    for i in range(xl-1):
        # make a vector to thicken our line points
        dx = x[i+1]-x[i]
        dy = y[i+1]-y[i]
        perp = np.array([-dy, dx])
        unit_perp = (perp/np.linalg.norm(perp))*linewidth

        # need to make 4 points for quadrilateral
        xs[i] = [x[i], x[i] + unit_perp[0]]
        ys[i] = [y[i], y[i] + unit_perp[1]]
        xs[i+1] = [x[i+1], x[i+1] + unit_perp[0]]
        ys[i+1] = [y[i+1], y[i+1] + unit_perp[1]]

        if len(z) == i+1:
            z.append(z[-1] + (dx**2+dy**2)**0.5)
        # set z values
        zs[i] = [z[i], z[i]]
        zs[i+1] = [z[i+1], z[i+1]]

    cm = plt.get_cmap(MAP)
    ax.pcolormesh(xs, ys, zs, shading='gouraud', cmap=cm)


'''make some path'''
np.random.seed(0)
n_dim, n_point = 10, 5
end_scale = 1
noise_scale = 0.01
normalize = True
normalizer = 1
dynamic = True

n_path = 100
P = np.zeros((n_path, n_point, n_dim))
for i in range(n_path):
    P[i, :, :] = sample_context_drift(
        n_dim, n_point,
        end_scale=end_scale,
        noise_scale=noise_scale,
        normalize=normalize,
        normalizer=normalizer,
        dynamic=dynamic,
    )

f, ax = plt.subplots(1, 1, figsize=(7, 7))
for i in range(n_path):
    x, y = P[i, :, 0], P[i, :, 1]
    colored_line(x, y, linewidth=.005)
ax.axhline(0, linestyle='--', color='grey', alpha=.5)
ax.axvline(0, linestyle='--', color='grey', alpha=.5)

ax.set_title('Some random walks')
ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
sns.despine()

# similarity tensor
rst = np.zeros((n_point, n_path, n_path))
for t in range(n_point):
    P_t = P[:, t, :]
    rst[t] = np.corrcoef(P_t)

for t in np.arange(1, n_point):
    f, ax = plt.subplots(1, 1, figsize=(9, 7))
    sns.heatmap(rst[t], cmap='viridis', square=True, ax=ax)
    ax.set_title(f't={t}')
    # sns.clustermap(rst[t], cmap='viridis', square=True)

# similarity distribution
off_diag_rs = rst[-1][np.tril_indices(n_path, k=-1)]
f, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.distplot(off_diag_rs, ax=ax, norm_hist=True)
ax.set_title(f'average inter-context similarity, t={t}')
ax.set_xlabel('r')
ax.set_ylabel('freq')
ax.set_xlabel('r')
sns.despine()
