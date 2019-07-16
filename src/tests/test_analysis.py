# import warnings
import numpy as np
from task import SequenceLearning
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def test_compute_event_similarity_matrix():
    from analysis import compute_event_similarity_matrix
    n_param = np.random.randint(low=2, high=15)
    n_branch = np.random.randint(low=2, high=5)
    n_parts = 1
    task = SequenceLearning(
        n_param=n_param, n_branch=n_branch, n_parts=n_parts,
        pad_len=0, p_rm_ob_enc=0, p_rm_ob_rcl=0,
    )
    # gen samples
    n_samples = 10
    X, Y, misc = task.sample(n_samples, to_torch=False, return_misc=True)
    Y_int = np.array([misc[i][1] for i in range(n_samples)])

    sm_from_3d = compute_event_similarity_matrix(Y)
    sm_from_2d = compute_event_similarity_matrix(Y_int)
    assert np.all(sm_from_3d == sm_from_2d), \
        'RSA result should be consistent'


if __name__ == "__main__":
    n_iters = 3
    for _ in range(n_iters):
        test_compute_event_similarity_matrix()
    print("Everything passed")
