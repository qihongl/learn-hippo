import numpy as np
from task.StimSampler import StimSampler
from task.SequenceLearning import SequenceLearning


def test_task_stim_sampler():
    n_param = np.random.randint(low=2, high=15)
    n_branch = np.random.randint(low=2, high=5)
    stim_sampler = StimSampler(n_param, n_branch)
    states_vec, param_vals_vec, ctx, misc = stim_sampler._sample()
    # tests
    assert np.all(np.sum(states_vec, axis=1) == 1), \
        'check no empty state for all t'
    assert np.all(np.sum(param_vals_vec, axis=1) == 1), \
        'check no empty param val for all t'


def test_task_stim_sampler_inter_part_consistency():
    n_param = np.random.randint(low=2, high=15)
    n_branch = np.random.randint(low=2, high=5)
    n_parts = 2
    ss = StimSampler(n_param, n_branch)
    [states_vec, param_vals_vec], _ = ss.sample(n_parts)

    [o_sample_, q_sample_], [x_int, y_int] = ss.sample(n_parts)
    [o_keys_vec, o_vals_vec, o_ctxs_vec] = o_sample_
    [q_keys_vec, q_vals_vec, q_ctxs_vec] = q_sample_

    o_k_int = np.argmax(o_keys_vec, axis=-1)
    q_k_int = np.argmax(q_keys_vec, axis=-1)
    o_y_int = np.argmax(o_vals_vec, axis=-1)
    q_y_int = np.argmax(q_vals_vec, axis=-1)

    for ip in range(n_parts):
        assert np.all(q_k_int[ip] == x_int),\
            'different event parts should have the same SEQ of query keys'
        assert set(o_k_int[ip]) == set(x_int),\
            'different event parts should have the same SET of query keys'

        assert np.all(q_y_int[ip] == y_int),\
            'different event parts should have the same SEQ of query vals'
        for v in range(n_branch):
            assert np.sum(o_y_int[ip] == v) == np.sum(y_int == v),\
                'different event parts should have the same SET of observed vals'


def test_task_sequence_learning():
    '''test ordering k,v in x by time re-product y'''
    n_param = np.random.randint(low=2, high=15)
    n_branch = np.random.randint(low=2, high=5)
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
    n_iters = 3
    for _ in range(n_iters):
        test_task_stim_sampler()
        test_task_stim_sampler_inter_part_consistency()
        test_task_sequence_learning()
    print("Everything passed")
