import tensorflow as tf


def GaBP(A, b):


    def condition(U, U_old, P, l):
        sums = tf.reduce_sum((U - U_old) ** 2, axis=[1, 2])
        bool_vec = tf.greater_equal(sums, tf.ones_like(sums) * tf.constant(1e-06, dtype=tf.float64))
        return tf.reduce_all(bool_vec)


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


    zero = tf.constant(0., dtype=tf.float64)
    indices_nonzero = tf.where(tf.not_equal(A, zero))
    bool_offD = tf.tile(tf.not_equal(indices_nonzero[:, 0][:, None], indices_nonzero[:, 1][:, None]), [1, 2])
    indices_nonzero_offD = tf.reshape(tf.boolean_mask(indices_nonzero, bool_offD), [-1, 2])

    P = tf.diag(tf.diag_part(A))
    U = tf.matrix_diag(tf.transpose(tf.divide(b, tf.diag_part(A)[:, None])))
    n = tf.cast(tf.shape(indices_nonzero_offD)[0], dtype=tf.int64)

    U_old = tf.zeros_like(U)
    l = tf.constant(0, dtype=tf.int64)
    U_end, U_old_end, P_end, l_end = tf.while_loop(condition, GaBP_loop, (U, U_old, P, l), maximum_iterations=20)

    Pf = tf.reduce_sum(P_end, axis=0, keepdims=True)
    x = tf.reduce_sum(tf.multiply(U_end, P_end[None]), axis=1) / Pf
    return tf.transpose(x)  # , tf.transpose(Pf)