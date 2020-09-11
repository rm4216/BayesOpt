import tensorflow as tf


def GaBP(A, b):

    zero = tf.constant(0., dtype=tf.float64)
    indices_nonzero = tf.where(tf.not_equal(A, zero))
    bool_offD = tf.tile(tf.not_equal(indices_nonzero[:, 0][:, None], indices_nonzero[:, 1][:, None]), [1, 2])
    indices_nonzero_offD = tf.reshape(tf.boolean_mask(indices_nonzero, bool_offD), [-1, 2])

    P = tf.diag(tf.diag_part(A))
    # U = tf.diag(tf.divide(b, tf.diag_part(A)[:, None])[:, 0])
    U = tf.matrix_diag(tf.transpose(tf.divide(b, tf.diag_part(A)[:, None])))
    n = tf.cast(tf.shape(indices_nonzero_offD)[0], dtype=tf.int64)

    # sess = tf.Session()
    # _indices_nonzero = sess.run(indices_nonzero)
    # _bool_offD = sess.run(bool_offD)
    # _indices_nonzero_offD = sess.run(indices_nonzero_offD)
    # _P = sess.run(P)
    # _U = sess.run(U)
    # _n = sess.run(n)
    # import numpy as np
    # aD = np.array([2.06546501, 7.95879494, 7.88016085])[:, None]
    # D0 = b[:, 0][:, None] / aD
    # D1 = b[:, 1][:, None] / aD
    # _D0 = sess.run(D0)
    # _D1 = sess.run(D1)

    def condition(U, U_old, P, l):
        sums = tf.reduce_sum((U - U_old) ** 2, axis=[1, 2])
        bool_vec = tf.greater_equal(sums, tf.ones_like(sums) * tf.constant(1e-06, dtype=tf.float64))
        return tf.reduce_all(bool_vec)
        # return tf.greater_equal(tf.reduce_sum((U - U_old) ** 2), tf.constant(1e-06, dtype=tf.float64))


    def condition_ij(U, index, P):
        return tf.less(index, n)


    def update_ij(U, index, P):
        i = indices_nonzero_offD[index, 0]
        j = indices_nonzero_offD[index, 1]

        p_i_minus_j = tf.reduce_sum(P[:, i]) - P[j, i]
        tf.Assert(tf.not_equal(p_i_minus_j, 0.), [p_i_minus_j])
        Pij = - A[i, j] ** 2 / p_i_minus_j  # assuming A is symmetric

        range_vec = tf.range(tf.cast(tf.shape(P)[0], dtype=tf.int64), dtype=tf.int64)
        bool_i = tf.tile(tf.equal(range_vec, tf.ones_like(range_vec, dtype=tf.int64) * i)[:, None], [1, tf.shape(P)[0]])
        bool_j = tf.tile(tf.equal(range_vec, tf.ones_like(range_vec, dtype=tf.int64) * j)[None], [tf.shape(P)[0], 1])
        P = tf.where(tf.logical_and(bool_i, bool_j), tf.ones_like(P) * Pij, P)

        # h_i_minus_j = (tf.reduce_sum(tf.multiply(P[:, i][:, None], U[:, i][:, None])) - P[j, i] * U[j, i]) / p_i_minus_j
        h_i_minus_j = (tf.reduce_sum(tf.multiply(P[:, i][None, :, None], U[:, :, i][:, :, None]), axis=1) -
                       P[j, i] * U[:, j, i][:, None]) / p_i_minus_j

        Uij = - A[i, j] * h_i_minus_j / P[i, j]

        bi_tile = tf.tile(tf.expand_dims(bool_i, axis=0), [tf.shape(U)[0], 1, 1])
        bj_tile = tf.tile(tf.expand_dims(bool_j, axis=0), [tf.shape(U)[0], 1, 1])
        U = tf.where(tf.logical_and(bi_tile, bj_tile), tf.multiply(tf.expand_dims(Uij, axis=-1), tf.ones_like(U)), U)

        return U, index + 1, P


    def GaBP_loop(U, U_old, P, l):
        index = tf.constant(0, dtype=tf.int64)
        U_out, index_out, P_out = tf.while_loop(condition_ij, update_ij, (U, index, P))
        return U_out, U, P_out, l + 1


    U_old = tf.zeros_like(U)
    l = tf.constant(0, dtype=tf.int64)
    U_end, U_old_end, P_end, l_end = tf.while_loop(condition, GaBP_loop, (U, U_old, P, l), maximum_iterations=20)
    # sess = tf.Session()
    # _U_end = sess.run(U_end)
    # _U_old_end = sess.run(U_old_end)
    # _P_end = sess.run(P_end)
    # _l_end = sess.run(l_end)

    Pf = tf.reduce_sum(P_end, axis=0, keepdims=True)
    # x = tf.reduce_sum(tf.multiply(U_end, P_end), axis=0, keepdims=True) / Pf
    x = tf.reduce_sum(tf.multiply(U_end, P_end[None]), axis=1) / Pf
    # _Pf = sess.run(Pf)      # diagonal inverse
    # _x = sess.run(x)        # vector solve
    return tf.transpose(x), tf.transpose(Pf)
    # _U_out = sess.run(U_out)
    # _index_out = sess.run(index_out)
    # _P_out = sess.run(P_out)

    # i = indices[index, 0]
    # j = indices[index, 1]
    #
    # p_i_minus_j = tf.reduce_sum(P[:, i]) - P[j, i]
    # tf.Assert(tf.not_equal(p_i_minus_j, 0.), [p_i_minus_j])
    # Pij = - A[i, j] ** 2 / p_i_minus_j  # assuming A is symmetric
    #
    # range_vec = tf.range(tf.cast(tf.shape(P)[0], dtype=tf.int64), dtype=tf.int64)
    # bool_i = tf.tile(tf.equal(range_vec, tf.ones_like(range_vec, dtype=tf.int64)*i)[:, None], [1, tf.shape(P)[0]])
    # bool_j = tf.tile(tf.equal(range_vec, tf.ones_like(range_vec, dtype=tf.int64)*j)[None], [tf.shape(P)[0], 1])
    # P = tf.where(tf.logical_and(bool_i, bool_j), tf.ones_like(P) * Pij, P)
    #
    # h_i_minus_j = (tf.reduce_sum(tf.multiply(P[:, i][:, None], U[:, i][:, None])) - P[j, i] * U[j, i]) / p_i_minus_j
    # Uij = - A[i, j] * h_i_minus_j / P[i, j]
    #
    # U = tf.where(tf.logical_and(bool_i, bool_j), tf.ones_like(U) * Uij, U)    # check shapes
    #
    # _i = sess.run(i)
    # _j = sess.run(j)
    # _p_i_minus_j = sess.run(p_i_minus_j)
    # _Pij = sess.run(Pij)
    # _bool_i = sess.run(bool_i)
    # _bool_j = sess.run(bool_j)
    # _Pnew = sess.run(P)
    # _h_i_minus_j = sess.run(h_i_minus_j)
    # _Uij = sess.run(Uij)
    # _Unew = sess.run(U)


# A = tf.constant([[1.0000,    0.0,    0.1000],
#                  [0.0,    1.0000,    0.1000],
#                  [0.1000,    0.1000,    1.0000]], dtype=tf.float64)
A = tf.constant([[ 2.06546501, -1.26225196,  0.44911404],
                 [-1.26225196,  7.95879494,  1.68844008],
                 [ 0.44911404,  1.68844008,  7.88016085]], dtype=tf.float64)
# b = tf.constant([[1.], [1.], [1.]], dtype=tf.float64)
# b = tf.constant([[ 0.28053784],
#                  [-1.52577847],
#                  [ 0.43222881]], dtype=tf.float64)

b = tf.constant([[ 0.28053784, 1.],
                 [-1.52577847, 1.],
                 [ 0.43222881, 1.]], dtype=tf.float64)

# zero = tf.constant(0., dtype=tf.float64)
# indices_nonzero = tf.where(tf.not_equal(A, zero))
# bool_offD = tf.tile(tf.not_equal(indices_nonzero[:, 0][:, None], indices_nonzero[:, 1][:, None]), [1, 2])
# indices_nonzero_offD = tf.reshape(tf.boolean_mask(indices_nonzero, bool_offD), [-1, 2])
#
# P = tf.diag(tf.diag_part(A))
# U = tf.matrix_diag(tf.transpose(tf.divide(b, tf.diag_part(A)[:, None])))
# n = tf.cast(tf.shape(indices_nonzero_offD)[0], dtype=tf.int64)
#
# sess = tf.Session()
# _indices_nonzero = sess.run(indices_nonzero)
# _bool_offD = sess.run(bool_offD)
# _indices_nonzero_offD = sess.run(indices_nonzero_offD)
# _P = sess.run(P)
# _U = sess.run(U)
# _n = sess.run(n)
# import numpy as np
# aD = np.array([2.06546501, 7.95879494, 7.88016085])[:, None]
# D0 = b[:, 0][:, None] / aD
# # D1 = b[:, 1][:, None] / aD
# _D0 = sess.run(D0)
# # _D1 = sess.run(D1)
#
# U_old = tf.zeros_like(U)
# l = tf.constant(0, dtype=tf.int64)
# index = tf.constant(0, dtype=tf.int64)
#
# i = indices_nonzero_offD[index, 0]
# j = indices_nonzero_offD[index, 1]
#
# p_i_minus_j = tf.reduce_sum(P[:, i]) - P[j, i]
# tf.Assert(tf.not_equal(p_i_minus_j, 0.), [p_i_minus_j])
# Pij = - A[i, j] ** 2 / p_i_minus_j  # assuming A is symmetric
#
# range_vec = tf.range(tf.cast(tf.shape(P)[0], dtype=tf.int64), dtype=tf.int64)
# bool_i = tf.tile(tf.equal(range_vec, tf.ones_like(range_vec, dtype=tf.int64) * i)[:, None], [1, tf.shape(P)[0]])
# bool_j = tf.tile(tf.equal(range_vec, tf.ones_like(range_vec, dtype=tf.int64) * j)[None], [tf.shape(P)[0], 1])
# P = tf.where(tf.logical_and(bool_i, bool_j), tf.ones_like(P) * Pij, P)
#
# # h_i_minus_j = (tf.reduce_sum(tf.multiply(P[:, i][:, None], U[:, i][:, None])) - P[j, i] * U[j, i]) / p_i_minus_j
# h_i_minus_j = (tf.reduce_sum(tf.multiply(P[:, i][None, :, None], U[:, :, i][:, :, None]), axis=1) -
#                P[j, i] * U[:, j, i][:, None]) / p_i_minus_j
#
# # _h_i_minus_j = sess.run(h_i_minus_j)
# Uij = - A[i, j] * h_i_minus_j / P[i, j]
# # _Uij = sess.run(Uij)
#
# bi_tile = tf.tile(tf.expand_dims(bool_i, axis=0), [tf.shape(U)[0], 1, 1])
# bj_tile = tf.tile(tf.expand_dims(bool_j, axis=0), [tf.shape(U)[0], 1, 1])
# U = tf.where(tf.logical_and(bi_tile, bj_tile), tf.multiply(tf.expand_dims(Uij, axis=-1), tf.ones_like(U)), U)
#
# # _Unew = sess.run(U)
# # _Unew0 = _Unew[0, :, :]
# # _Unew1 = _Unew[1, :, :]
# sums = tf.reduce_sum((U - U_old) ** 2, axis=[1,2])
# bool_vec = tf.greater_equal(sums, tf.ones_like(sums) * tf.constant(1e-06, dtype=tf.float64))
# all_s = tf.reduce_all(bool_vec)
#
# # _sums = sess.run(sums)
# # _bool_vec = sess.run(bool_vec)
# # _all_s = sess.run(all_s)

x, Pf = GaBP(A, b)
sess = tf.Session()
_x = sess.run(x)
_Pf = sess.run(Pf)
