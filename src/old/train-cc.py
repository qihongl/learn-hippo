"""demo: train a DND LSTM on a contextual choice task
"""
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models.LCALSTM_v9 import LCALSTM as Agent
# from models import LCALSTM as Agent
from task import ContextualChoice
from analysis import entropy, compute_stats
# from models.DND import compute_similarities
from models import get_reward, compute_returns, compute_a2c_loss
from task.utils import get_one_hot_vector
from utils.utils import to_pth, to_sqnp, to_np
from sklearn.decomposition import PCA
from scipy.stats import sem

sns.set(style='white', context='talk', palette='colorblind')
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)

# env param
penalty = 1

# gen training set
# n time steps of a trial
trial_length = 10
# after `tp_corrupt`, turn off the noise
t_noise_off = trial_length // 2
# input/output/hidden/memory dim
obs_dim = 32
task = ContextualChoice(
    obs_dim=obs_dim, trial_length=trial_length, t_noise_off=t_noise_off
)

# num unique training examples in one epoch
n_unique_example = 5
X, Y = task.sample(n_unique_example)
n_trials = len(X)
print(f'X.size: {X.size()}, n_trials x trial_length x x-dim')
print(f'Y.size: {Y.size()}, n_trials x trial_length x y-dim')


# set params
dim_hidden = 64
dim_hidden_dec = 32
dim_output = 2

dict_len = 100
learning_rate = 1e-3
n_epochs = 300
eta = 0

# init model and hidden state.
agent = Agent(
    input_dim=task.x_dim, output_dim=dim_output,
    rnn_hidden_dim=dim_hidden, dec_hidden_dim=dim_hidden_dec,
    dict_len=dict_len, kernel='cosine',
)
# agent = Agent(task.x_dim, dim_hidden, dim_output, dict_len=dict_len)
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=1/2, patience=50, threshold=1e-3, min_lr=1e-8,
    verbose=True
)

'''train'''
log_return = np.zeros(n_epochs,)
log_ent = np.zeros(n_epochs,)
log_loss_value = np.zeros(n_epochs,)
log_loss_policy = np.zeros(n_epochs,)

log_Y = np.zeros((n_epochs, n_trials, trial_length))
log_Y_hat = np.zeros((n_epochs, n_trials, trial_length))

log_cache = [[[None] for _ in range(trial_length)] for _ in range(n_trials)]

# loop over epoch
i, m, t = 0, 0, 0
for i in range(n_epochs):
    time_start = time.time()
    # get data for this epoch
    X, Y = task.sample(n_unique_example)
    # flush hippocampus
    agent.flush_episodic_memory()

    # loop over the training set
    for m in range(n_trials):
        # prealloc
        cumulative_reward = 0
        cumulative_entropy = 0
        probs, rewards, values = [], [], []
        hc_t = agent.get_init_states()

        if m > n_unique_example:
            agent.retrieval_on()
        else:
            agent.retrieval_off()

        # loop over time, for one training example
        for t in range(trial_length):
            y_oh_m_t = to_pth(get_one_hot_vector(Y[m][t], 2))
            # only save memory at the last time point
            agent.encoding_off()
            if t == trial_length-1 and m < n_unique_example:
                agent.encoding_on()
            # recurrent computation at time t
            pi_a_t, value_t, (h_t, cm_t), cache = agent(
                X[m][t].view(1, 1, -1), hc_t)
            # action selection
            a_t, prob_a_t = agent.pick_action(pi_a_t)
            # compute immediate reward
            r_t = get_reward(a_t, y_oh_m_t, penalty=penalty)
            # compute response entropy
            cumulative_entropy += entropy(pi_a_t)
            # log
            probs.append(prob_a_t)
            rewards.append(r_t)
            values.append(value_t)
            # log
            cumulative_reward += r_t
            log_Y_hat[i, m, t] = a_t.item()
            log_cache[m][t] = cache

        returns = compute_returns(rewards, normalize=True)
        loss_policy, loss_value = compute_a2c_loss(probs, values, returns)
        loss = loss_policy + loss_value - eta * cumulative_entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        log_Y[i] = np.squeeze(Y.numpy())
        log_ent[i] = cumulative_entropy / n_trials
        log_return[i] += cumulative_reward / n_trials
        log_loss_value[i] += loss_value.item() / n_trials
        log_loss_policy[i] += loss_policy.item() / n_trials

    scheduler.step(log_return[i])
    # print out some stuff
    time_end = time.time()
    run_time = time_end - time_start
    print(
        'Epoch %3d | return = %.2f, ent = %.2f | loss: val = %.2f, pol = %.2f | time = %.2f' %
        (i, log_return[i], log_ent[i],
         log_loss_value[i], log_loss_policy[i], run_time)
    )


'''org data'''
# network internal reps
inpt = np.full((n_trials, trial_length), np.nan)
leak = np.full((n_trials, trial_length), np.nan)
comp = np.full((n_trials, trial_length), np.nan)
C = np.full((n_trials, trial_length, dim_hidden), np.nan)
H = np.full((n_trials, trial_length, dim_hidden), np.nan)
M = np.full((n_trials, trial_length, dim_hidden), np.nan)
CM = np.full((n_trials, trial_length, dim_hidden), np.nan)
DA = np.full((n_trials, trial_length, dim_hidden_dec), np.nan)
V = [None] * n_trials

for i in range(n_trials):
    for t in range(trial_length):
        # unpack data for i,t
        [vector_signal, scalar_signal, misc] = log_cache[i][t]
        [inpt_it, leak_it, comp_it] = scalar_signal
        [h_t, m_t, cm_t, des_act_t, V_i] = misc
        # cache data to np array
        inpt[i, t] = to_sqnp(inpt_it)
        leak[i, t] = to_sqnp(leak_it)
        comp[i, t] = to_sqnp(comp_it)
        H[i, t, :] = to_sqnp(h_t)
        M[i, t, :] = to_sqnp(m_t)
        CM[i, t, :] = to_sqnp(cm_t)
        DA[i, t, :] = to_sqnp(des_act_t)
        V[i] = V_i

# compute cell state
C = CM - M

'''analysis'''
f, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
axes[0, 0].plot(log_return)
axes[0, 0].set_ylabel('Return')

axes[0, 1].plot(log_ent)
axes[0, 1].set_ylabel('Ent')

axes[1, 0].plot(log_loss_value)
axes[1, 0].set_ylabel('Value loss')

axes[1, 1].plot(log_loss_policy)
axes[1, 1].set_ylabel('Policy loss')

axes[1, 1].set_xlabel('Epoch')
axes[1, 0].set_xlabel('Epoch')

sns.despine()
f.tight_layout()

'''prediction accuracy'''
n_se = 2
# compute stat
corrects = log_Y_hat[-1] == log_Y[-1]
mu_mem0 = np.mean(corrects[:n_unique_example], axis=0)
er_mem0 = sem(corrects[:n_unique_example], axis=0) * n_se
mu_mem1 = np.mean(corrects[n_unique_example:], axis=0)
er_mem1 = sem(corrects[n_unique_example:], axis=0) * n_se

f, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.errorbar(range(trial_length), y=mu_mem0, yerr=er_mem0, label='w/o memory')
ax.errorbar(range(trial_length), y=mu_mem1, yerr=er_mem1, label='w/  memory')
ax.axvline(t_noise_off, label='turn off noise', color='grey', linestyle='--')
# ax.axhline(1/2, color='grey', linestyle='--')
ax.set_xlabel('Time')
ax.set_ylabel('Correct rate')
ax.set_title('Behavioral signature of memory based choice')
f.legend(frameon=False, bbox_to_anchor=(.9, .6))
f.tight_layout()
sns.despine()

# f.savefig('../figs/correct-rate.png', dpi=100, bbox_inches='tight')

'''lca params'''
inpt_nr_mu, inpt_nr_er = compute_stats(inpt[:n_unique_example])
inpt_r_mu, inpt_r_er = compute_stats(inpt[n_unique_example:])
leak_nr_mu, leak_nr_er = compute_stats(leak[:n_unique_example])
leak_r_mu, leak_r_er = compute_stats(leak[n_unique_example:])
comp_nr_mu, comp_nr_er = compute_stats(comp[:n_unique_example])
comp_r_mu, comp_r_er = compute_stats(comp[n_unique_example:])

f, axes = plt.subplots(3, 1, figsize=(7, 8))
axes[0].errorbar(x=range(trial_length), y=inpt_r_mu, yerr=inpt_r_er)
axes[0].errorbar(x=range(trial_length), y=inpt_nr_mu, yerr=inpt_nr_er)
axes[1].errorbar(x=range(trial_length), y=leak_r_mu, yerr=leak_r_er)
axes[1].errorbar(x=range(trial_length), y=leak_nr_mu, yerr=leak_nr_er)
axes[2].errorbar(x=range(trial_length), y=comp_r_mu, yerr=comp_r_er)
axes[2].errorbar(x=range(trial_length), y=comp_nr_mu, yerr=comp_nr_er)
sns.despine()
f.tight_layout()