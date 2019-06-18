import numpy as np
from task.MovieSampler import MovieSampler


def test_task_stim_sampler():
    n_param, n_branch = 7, 3
    n_timesteps = n_param
    stim_sampler = MovieSampler(n_param, n_branch)
    states_vec, param_vals_vec = stim_sampler._sample(n_timesteps)
    # tests
    assert np.all(np.sum(states_vec, axis=1) == 1), \
        'check no empty state for all t'
    assert np.all(np.sum(param_vals_vec, axis=1) == 1), \
        'check no empty param val for all t'


def test_task_stim_sampler_inter_part_consistency():
    n_param, n_branch = 7, 3
    n_timesteps = n_param
    n_parts = 2
    es = MovieSampler(n_param, n_branch)
    [states_vec, param_vals_vec], _ = es.sample(
        n_timesteps, n_parts, xy_format=False
    )
    s_feature_vec = np.sum(states_vec[0], axis=0)
    p_counts_vec = np.sum(param_vals_vec[0], axis=0)
    for ip in np.arange(1, n_parts):
        assert np.all(
            np.sum(states_vec[ip], axis=0) == s_feature_vec
        ), 'different event parts should have the same set of states'
        assert np.all(
            np.sum(param_vals_vec[ip], axis=0) == p_counts_vec
        ), 'different event parts should have the same set of parameter values'


if __name__ == "__main__":
    test_task_stim_sampler()
    test_task_stim_sampler_inter_part_consistency()
    print("Everything passed")
