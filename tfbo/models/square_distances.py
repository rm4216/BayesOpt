import numpy as np


def square_dists_np(X1, X2):
    X1_square = np.diag(np.matmul(X1, X1.transpose()))[:, None]
    X2_square = np.diag(np.matmul(X2, X2.transpose()))[None]
    X1X2_dp = 2 * np.matmul(X1, X2.transpose())
    pw_square_dists = X1_square + X2_square - X1X2_dp
    return pw_square_dists  # test it