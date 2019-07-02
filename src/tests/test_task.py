import numpy as np
from task.StimSampler import StimSampler
from task.SequenceLearning import SequenceLearning


def test_task_stim_sampler():
    n_param, n_branch = 7, 3
    stim_sampler = StimSampler(n_param, n_branch)
    states_vec, param_vals_vec, ctx = stim_sampler._sample()
    # tests
    assert np.all(np.sum(states_vec, axis=1) == 1), \
        'check no empty state for all t'
    assert np.all(np.sum(param_vals_vec, axis=1) == 1), \
        'check no empty param val for all t'


def test_task_stim_sampler_inter_part_consistency():
    n_param, n_branch = 7, 3
    n_parts = 2
    es = StimSampler(n_param, n_branch)
    [states_vec, param_vals_vec, _], _ = es.sample(n_parts)
    s_feature_vec = np.sum(states_vec[0], axis=0)
    p_counts_vec = np.sum(param_vals_vec[0], axis=0)
    for ip in np.arange(1, n_parts):
        assert np.all(
            np.sum(states_vec[ip], axis=0) == s_feature_vec
        ), 'different event parts should have the same set of states'
        assert np.all(
            np.sum(param_vals_vec[ip], axis=0) == p_counts_vec
        ), 'different event parts should have the same set of parameter values'


def test_task_sequence_learning():
    '''test ordering k,v in x by time re-product y'''
    n_param, n_branch = 6, 5
    n_parts = 2
    n_samples = 3
    sl = SequenceLearning(n_param, n_branch, n_parts=n_parts)
    X, Y = sl.sample(n_samples, to_torch=False)
    for i in range(n_samples):
        x, y = X[i], Y[i]
        # test
        T_part = n_param
        time_ids = np.argmax(x[:T_part, :sl.k_dim], axis=1)
        sort_ids = np.argsort(time_ids)
        x_sorted = x[:T_part, :sl.k_dim][sort_ids]
        y_sorted = x[:T_part, sl.k_dim:sl.k_dim+sl.v_dim][sort_ids]
        assert np.all(x_sorted == np.eye(n_param))
        assert np.all(y_sorted == y[:T_part])


if __name__ == "__main__":
    test_task_stim_sampler()
    test_task_stim_sampler_inter_part_consistency()
    test_task_sequence_learning()
    print("Everything passed")
