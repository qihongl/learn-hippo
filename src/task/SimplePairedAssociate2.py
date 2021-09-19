import numpy as np
from utils.utils import to_pth


class SimplePairedAssociate2():
    '''
    a simple paired associate task with 2 time steps
    note:
    each cue
    - has a "schematic" associate that occurs 75% of the time;
    - the other 25% of the time it's paired with a randomly generated (non-schematic) associate.

    the "selective encoding" policy will be:
    - when the schematic associate occurs at study, don't encode;
    - when a non-schematic associate occurs at study, encode.
    - note: applying this policy will result in a 3/4 reduction in the number of stored memory traces,
    which should reduce confusion a lot, especially when the cues are very similar.

    Experiment procedure
    study:  x = [cue, _____], y = [assoc] *the only difference from the prev version
            x = [cue, assoc], y = [assoc]
    test :  x = [cue, None ], y = [assoc]
    '''

    def __init__(self, n_cue=16, n_assoc=32, schema_level=.5):
        self.n_cue = n_cue
        self.n_assoc = n_assoc
        self.schema_level = schema_level
        self.reset()

    def reset(self):
        # init cue/assoc representation to one hot vectors
        self.Cue = np.eye(self.n_cue)
        # self.Cue = np.ones((self.n_cue, self.n_cue)) - np.eye(self.n_cue)
        self.Assoc = np.eye(self.n_assoc)
        self.dim_cue = self.n_cue
        self.dim_assoc = self.n_assoc
        self.x_dim = self.n_cue + self.n_assoc
        self.y_dim = self.n_assoc
        # form a pairing between cue and assoc
        self.cue, self.assoc = self._form_pairing()

    def _form_pairing(self):
        '''
        generate #n_cue trials
        '''
        cue = self.Cue
        assoc = np.zeros((self.n_cue, self.dim_assoc))
        id_used_assocs = []
        n_schematic = self.schema_level * self.n_cue
        schematic_ids = np.random.choice(
            np.arange(self.n_cue), size=int(n_schematic), replace=False
        )

        for i in range(self.n_cue):
            if i in schematic_ids:
                # for schematic trial, pair assoc i with cue i (matching index)
                assoc[i, :] = self.Assoc[i, :]
            else:
                # randomly pick an associate, whose index > n_cue
                while True:
                    j = np.random.choice(np.arange(self.n_cue, self.n_assoc))
                    if j not in id_used_assocs:
                        id_used_assocs.append(j)
                        # print(i, j, id_used_assocs)
                        break
                assoc[i, :] = self.Assoc[j, :]
        return cue, assoc

    # def permute_cue_assoc(self):
    #     perm = np.random.permutation(np.arange(self.n_cue))
    #     cue_perm = np.vstack([self.cue[i, :] for i in perm])
    #     assoc_perm = np.vstack([self.assoc[i, :] for i in perm])
    #     return cue_perm, assoc_perm, perm

    def sample_cue_assoc(self, permute=False):
        if permute:
            perm = np.random.permutation(np.arange(self.n_cue))
        else:
            perm = np.arange(self.n_cue)
        cue_perm = np.vstack([self.cue[i, :] for i in perm])
        assoc_perm = np.vstack([self.assoc[i, :] for i in perm])
        return cue_perm, assoc_perm, perm

    def compute_schematic(self, cue_, assoc_):
        cue_ids = np.argmax(cue_, axis=1)
        # cue_ids = np.argmin(cue_, axis=1)
        assoc_ids = np.argmax(assoc_, axis=1)
        schematic = cue_ids == assoc_ids
        return schematic

    def sample(self, return_misc=True, to_torch=False):
        # construct study phase order and test phase order for the pairing
        cue_std, assoc_std, order_std = self.sample_cue_assoc(permute=True)
        cue_tst, assoc_tst, order_tst = self.sample_cue_assoc(permute=True)
        # whether each trial is schematic
        schematic_std = self.compute_schematic(cue_std, assoc_std)
        schematic_tst = self.compute_schematic(cue_tst, assoc_tst)
        # form X and Y
        # study:    x = [cue, assoc], y = [assoc]
        # study:    x = [cue, _____], y = [assoc] *the only difference from the prev version
        #           x = [cue, assoc], y = [assoc]
        X_std_t1 = np.hstack([cue_std, np.zeros(np.shape(assoc_std))])
        X_std_t2 = np.hstack([cue_std, assoc_std])
        Y_std_t1 = np.hstack([assoc_std])
        X_std = interweaving2arrays(X_std_t1, X_std_t2)
        Y_std = interweaving2arrays(Y_std_t1, Y_std_t1)
        # f, axes = plt.subplots(1, 2, figsize=(12, 12))
        # axes[0].imshow(X_std)
        # axes[1].imshow(Y_std)

        # test :    x = [cue, None ], y = [assoc]
        X_tst = np.hstack([cue_tst, np.zeros(np.shape(assoc_tst))])
        Y_tst = np.hstack([assoc_tst])
        # combine study and test phase data
        X = np.vstack([X_std, X_tst])
        Y = np.vstack([Y_std, Y_tst])
        schematic = np.stack([schematic_std, schematic_tst])
        order = np.stack([order_std, order_tst])

        if to_torch:
            X, Y = to_pth(X), to_pth(Y)
        if return_misc:
            return X, Y, [schematic, order]
        return X, Y


def interweaving2arrays(a, b):
    '''
    merge 2 arrays by interleave between rows from that 2 array
    ref:https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays

    example:
    a = np.zeros((3, 2))
    b = np.ones((3, 2))
    c = interweaving2arrays(a, b)
    print(c)
    '''
    assert np.shape(a) == np.shape(b)
    c = np.zeros((np.shape(a)[0] + np.shape(b)[0], np.shape(a)[1]))
    c[0::2, :] = a
    c[1::2, :] = b
    return c


'''test'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    # np.random.seed(0)
    sns.set(style='white', palette='colorblind', context='poster')
    n_cue = 16
    n_assoc = 32
    schema_level = .5
    spa = SimplePairedAssociate2(
        n_cue=n_cue, n_assoc=n_assoc, schema_level=schema_level
    )

    '''1. show cue and assoc pairing'''
    cue, assoc = spa._form_pairing()
    # f, axes = plt.subplots(
    #     1, 2, figsize=(12, 8),
    #     gridspec_kw={'width_ratios': [spa.dim_cue, spa.dim_assoc]}
    # )
    # axes[0].imshow(cue)
    # axes[1].imshow(assoc)
    # axes[0].set_xlabel('dim')
    # axes[1].set_xlabel('dim')
    # axes[0].set_ylabel('Trials')
    # axes[0].set_title('Cue')
    # axes[1].set_title('Associate')
    # f.tight_layout()

    '''2. show sample'''
    X, Y, schematic = spa.sample()

    f, axes = plt.subplots(
        2, 2, figsize=(10, 14), sharey=True,
        gridspec_kw={
            'width_ratios': [spa.dim_cue, spa.dim_assoc],
            'height_ratios': [len(X), len(X)]
        }
    )

    cue, assoc = X[:, :spa.dim_cue], X[:, spa.dim_cue:]

    axes[0, 0].imshow(cue)
    axes[0, 1].imshow(assoc)
    axes[1, 1].imshow(Y)
    axes[1, 0].imshow(cue)
    axes[0, 1].axvline(n_cue - 1 / 2, color='grey', linestyle='--')
    axes[1, 1].axvline(n_cue - 1 / 2, color='grey', linestyle='--')
    for ax in axes.reshape(-1):
        ax.axhline(n_cue * 2 - 1 / 2, color='red', linestyle='--')
    axes[0, 0].set_ylabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title('FOR REF ONLY')
    axes[1, 0].set_xlabel('Feature dim')
    axes[1, 1].set_xlabel('Feature dim')
    axes[0, 0].set_title('Cue')
    axes[0, 1].set_title('Assoc')
    f.tight_layout()

    '''2. show sample - for the paper'''

    f, axes = plt.subplots(
        2, 3, figsize=(14, 9), sharey='row', sharex='col',
        gridspec_kw={
            'width_ratios': [spa.dim_cue, spa.dim_assoc, spa.dim_assoc],
            'height_ratios': [spa.dim_cue * 2, spa.dim_cue],
        }
    )

    cue, assoc = X[:, :spa.dim_cue], X[:, spa.dim_cue:]
    cmap = 'bone'
    axes[0, 0].imshow(cue[:spa.dim_cue * 2, :], cmap=cmap)
    axes[0, 1].imshow(assoc[:spa.dim_cue * 2, :], cmap=cmap)
    axes[0, 2].imshow(Y[:spa.dim_cue * 2, :], cmap=cmap)

    axes[1, 0].imshow(cue[spa.dim_cue * 2:, :], cmap=cmap)
    axes[1, 1].imshow(assoc[spa.dim_cue * 2:, :], cmap=cmap)
    axes[1, 2].imshow(Y[spa.dim_cue * 2:, :], cmap=cmap)

    # for i in range(3):
    #     axes[1, i].set_xlabel('Feature dim')
    axes[1, 0].set_xticks(np.arange(0, spa.dim_cue, 5))
    axes[1, 1].set_xticks(np.arange(0, spa.dim_assoc, 10))
    axes[1, 2].set_xticks(np.arange(0, spa.dim_assoc, 10))
    axes[0, 0].set_ylabel('Study block', fontname='Helvetica')
    axes[1, 0].set_ylabel('Test block', fontname='Helvetica')
    axes[0, 0].set_title('Cue', fontname='Helvetica')
    axes[0, 1].set_title('Assoc', fontname='Helvetica')
    axes[0, 2].set_title('Assoc', fontname='Helvetica')
    f.tight_layout()
    f.savefig(f'examples/figs/stimulus-rep-spa2.png',
              dpi=100, bbox_inches='tight')
