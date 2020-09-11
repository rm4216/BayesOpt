import numpy as np
import tensorflow as tf
import gpflow
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.models.cov_funcs import TreeCoregion


def build_tree_indices():
    tree = []
    for i in range(D):
        tree.append([i, i])

    i = int(1)
    j = int(1)
    while(j < D - 1):
        tree.append([i - 1, i * 2 - 1])
        if (i * 2 - 1 < D - 1):
            tree.append([i - 1, i * 2])
        j = tree[-1][1]
        i = i + 1
    return tree


np.random.seed(123)
D = int(7)      # output dimension
X = np.concatenate([np.random.normal(loc=0., scale=1., size=[D, 1]), np.arange(D)[:, None]], axis=1)
X2 = np.concatenate([np.random.normal(loc=0., scale=1., size=[4, 1]), np.arange(4)[:, None]], axis=1)
binary_tree_list = build_tree_indices()
values = np.random.normal(loc=0., scale=1., size=[len(binary_tree_list)])
values_tanh = np.tanh(values)
values_comparison = values_tanh + np.concatenate([np.ones(D), np.zeros(len(binary_tree_list)-D)]) * D

tree_coregion = TreeCoregion(input_dim=int(1),
                             output_dim=D,
                             indices_tree=binary_tree_list,
                             values=values,
                             active_dims=[1])

# tree_coregion = gpflow.kernels.Coregion(input_dim=1, output_dim=D,
#                                 rank=D, active_dims=[1])
# tree_coregion.W = np.random.randn(D, D)

# indices_symm = tf.constant(binary_tree_list, dtype=tf.int64)
# values_symm = tf.constant(values, dtype=tf.float64)
# shape = tf.constant([D, D], dtype=tf.int64)
# B = tf.sparse_to_dense(sparse_indices=indices_symm, sparse_values=tf.tanh(values_symm), output_shape=[D, D], validate_indices=False)
# sess = tf.Session()
# _B = sess.run(B)
Kc = tree_coregion.compute_K_symm(X)
positivity = np.all(np.linalg.eigvals(Kc) > 0)
Kc0 = tree_coregion.compute_K(X, X2)
Kc1 = tree_coregion.compute_Kdiag(X)

# def build_tree_indices():
#     tree = []
#     for i in range(D):
#         tree.append([i, i])
#
#     i = int(1)
#     j = int(1)
#     while(j < D):
#         tree.append([i - 1, i * 2 - 1])
#         if (i * 2 - 1 < D):
#             tree.append([i - 1, i * 2])
#         j = tree[-1][1]
#         i = i + 1
#     return tree