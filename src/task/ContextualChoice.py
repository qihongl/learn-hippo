import torch
import numpy as np
from utils.utils import to_pth


class ContextualChoice():

    def __init__(self, obs_dim, trial_length=10, t_noise_off=5):
        self.obs_dim = obs_dim
        self.ctx_dim = obs_dim
        self.trial_length = trial_length
        self.t_noise_off = t_noise_off
        # 2nd level params
        self.x_dim = self.obs_dim + self.ctx_dim
        self.y_dim = 1
        # input validation
        assert t_noise_off < trial_length

    def sample(
            self, n_unique_examples,
            to_torch=True,
    ):
        """sample a task sequence

        Parameters
        ----------
        n_unique_examples : type
            Description of parameter `n_unique_examples`.
        to_torch : type
            Description of parameter `to_torch`.

        Returns
        -------
        type
            Description of returned object.

        """
        observation_p1, target_p1, context_p1 = self._sample_n_trials(
            n_unique_examples)
        # form the 2nd part of the data
        [observation_p2, target_p2, context_p2] = _permute_array_list(
            [observation_p1, target_p1, context_p1])
        # concat observation and context
        observation_context_p1 = np.dstack([observation_p1, context_p1])
        observation_context_p2 = np.dstack([observation_p2, context_p2])
        # combine the two phases
        X = np.vstack([observation_context_p1, observation_context_p2])
        Y = np.vstack([target_p1, target_p2])
        # to pytorch form
        if to_torch:
            X = to_pth(X)
            Y = to_pth(Y, pth_dtype=torch.LongTensor)
        return X, Y

    def _sample_n_trials(self, n_examples):
        observation = np.zeros((n_examples, self.trial_length, self.obs_dim))
        context = np.zeros((n_examples, self.trial_length, self.ctx_dim))
        target = np.zeros((n_examples, self.trial_length, self.y_dim))
        for i in range(n_examples):
            observation[i], context[i], target[i] = self._sample_one_trial()
        return observation, target, context

    def _sample_one_trial(self):
        """
        evidence:
            initially ambiguous,
            after `t_noise_off`, become predictive about the target
        """
        evidence = np.random.normal(
            loc=np.sign(np.random.normal()),
            size=(self.trial_length, self.obs_dim)
        )
        # integration results
        target_value = 1 if np.sum(evidence) > 0 else 0
        target = np.tile(target_value, (self.trial_length, 1))
        # corrupt the evidence input
        evidence[:self.t_noise_off] = np.random.normal(
            loc=0, size=(self.t_noise_off, self.obs_dim)
        )
        # generate a cue
        cue_t = np.random.normal(size=(self.ctx_dim, ))
        cue = np.tile(cue_t, (self.trial_length, 1))
        return evidence, cue, target


def _permute_array_list(input_list):
    """permute a list of n-d arrays

    Parameters
    ----------
    input_list : list
        a list of arrays, 0-th dim must be the same

    Returns
    -------
    list
        a list of arrays, permuted in the same way (along the 0-th dim)

    """
    n_examples_ = len(input_list[0])
    for np_array in input_list:
        assert np.shape(np_array)[0] == n_examples_, \
            f'{np.shape(np_array)} != n_examples_ == ({n_examples_})'
    perm_op = np.random.permutation(n_examples_)
    # for every list, permute them by
    perm_list = [input_list_j[perm_op] for input_list_j in input_list]
    return perm_list


'''how to use'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', context='talk')

    # build a sampler
    obs_dim = 20
    trial_length = 10
    t_noise_off = 5
    task = ContextualChoice(
        obs_dim=obs_dim,
        trial_length=trial_length,
        t_noise_off=t_noise_off
    )
    # sample
    n_examples = 5
    X, Y = task.sample(n_examples, to_torch=False)
    print(f'X shape = {np.shape(X)}, n_example x time x x-dim')
    print(f'Y shape = {np.shape(Y)},  n_example x time x y-dim')

    # show one trial
    i = 0
    input = X[i]
    target = int(Y[i][0])
    vmin = np.min(X[i])
    vmax = np.max(X[i])

    f, ax = plt.subplots(1, 1, figsize=(3, 5))
    sns.heatmap(
        input.T,
        vmin=vmin, vmax=vmax,
        cmap='RdBu_r', yticklabels=10, center=0,
        ax=ax
    )
    ax.axvline(t_noise_off, color='grey', linestyle='--')
    ax.axhline(obs_dim, color='black', linestyle='--')
    ax.set_title(f'Stimulus for a trial, y = {target}')
    ax.set_xlabel('Time')
    ax.set_ylabel('x-dim: context | input')
    f.savefig(f'examples/figs/eg-{target}.png', dpi=100, bbox_inches='tight')
