# import numpy as np
# from analysis.behav import compute_predacc
#
#
# def test_compute_correct_rate():
#     '''assert all correct, given Y = Yhat'''
#     n_examples, total_event_len, ohv_dim = 2, 3, 4
#     Y = Yhat = np.random.normal(size=(n_examples, total_event_len, ohv_dim))
#     corrects = compute_predacc(Y, Yhat)
#     assert np.sum(corrects) == n_examples * total_event_len
#
#
# if __name__ == "__main__":
#     print("Everything passed")
