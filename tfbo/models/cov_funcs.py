from gpflow import kernels
import tensorflow as tf
from gpflow.params import Parameter, Parameterized, ParamList
from gpflow.decors import params_as_tensors, autoflow
from gpflow import transforms
from gpflow import settings
import numpy as np
from gpflow.kernels import Kernel


class Collection(kernels.Combination):
    def K(self, X, X2=None, presliced=False, i=None):
        if i is None:
            return kernels.reduce(tf.add, [k.K(X, X2) for k in self.kernels])
        else:
            return kernels.reduce(tf.add, [self.kernels[index].K(X, X2) for index in i])

    def Kdiag(self, X, presliced=False, i=None):
        if i is None:
            return kernels.reduce(tf.add, [k.Kdiag(X) for k in self.kernels])
        else:
            return kernels.reduce(tf.add, [self.kernels[index].Kdiag(X) for index in i])


class Kstack(kernels.Combination):
    def K(self, X, X2=None, presliced=False, i=None):
        if isinstance(i, int):
            return self.kernels[i].K(X, X2)
        else:
            return tf.stack([self.kernels[index].K(X, X2) for index in i], axis=0)

    def Kdiag(self, X, presliced=False, i=None):
        if isinstance(i, int):
            return self.kernels[i].Kdiag(X)
        else:
            return tf.stack([self.kernels[index].Kdiag(X) for index in i], axis=0)


class Kparallel(kernels.Combination):
    def K(self, X, X2=None, presliced=False, i=None, base_coregion=None):
        '''

        :param X:
        :param X2:
        :param presliced:
        :param i: indicates the block of the dimensional decomposition e.g. i = 0,...,19
        :param base_coregion: indicates whether to evaluate the base kernel (0) or the coregionalization kernel (1) or indistinctively (2)
        :return:
        '''
        if base_coregion == int(2):
            if isinstance(i, int):
                return self.kernels[i].K(X, X2)
            else:
                return tf.stack([self.kernels[index].K(X, X2) for index in i], axis=0)
        elif base_coregion == int(1) or base_coregion == int(0):
            if isinstance(i, int):
                return self.kernels[i].kernels[base_coregion].K(X, X2)
            else:
                return tf.stack([self.kernels[index].kernels[base_coregion].K(X, X2) for index in i], axis=0)
        else:
            ValueError('base_coregion value not specified')
            return


    def Kdiag(self, X, presliced=False, i=None, base_coregion=None):
        # i: indicates the block of the dimensional decomposition e.g. i = 0,...,19
        # base_coregion: indicates whether to evaluate the base kernel (0) or the coregionalization kernel (1) or indistinctively (2)
        if base_coregion == int(2):
            if isinstance(i, int):
                return self.kernels[i].Kdiag(X)
            else:
                return tf.stack([self.kernels[index].Kdiag(X) for index in i], axis=0)
        elif base_coregion == int(1) or base_coregion == int(0):
            if isinstance(i, int):
                return self.kernels[i].kernels[base_coregion].Kdiag(X)
            else:
                return tf.stack([self.kernels[index].kernels[base_coregion].Kdiag(X) for index in i], axis=0)
        else:
            ValueError('base_coregion value not specified')
            return


class Multiple_k(kernels.Combination):
    '''
    Collection of kernels for joint Manifold GP and Full MOGP
    The model assumed for the MOGP is the Intrinsic Coregionalization Model, i.e. coregion kernel and a standard kernel
    The assumed order of the kernels are the following:
    kernels[0] = gpflow.kernels.Coregion
    kernels[1] = gpflow.kernels.Matern52/RBF/etc.
    kernels[2] = gpflow.kernels.Matern52/RBF/etc.
    '''
    def K(self, X, X2=None, presliced=False, i=None):
        # only accepts integers for "i"
        return self.kernels[i].K(X, X2)

    def Kdiag(self, X, presliced=False, i=None):
        return self.kernels[i].Kdiag(X)


class LinearGeneralized(Kernel):
    """
    The linear kernel
    """

    def __init__(self, input_dim, L_p=None, active_dims=None, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          if ARD=True, there is one variance per input
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        super().__init__(input_dim, active_dims, name=name)

        self.L_p = Parameter(np.eye(input_dim, dtype=settings.float_type),
                             dtype=settings.float_type) if L_p is None else Parameter(L_p, dtype=settings.float_type)

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        Sigma_p = tf.matmul(self.L_p, self.L_p, transpose_b=True)
        if X2 is None:
            return tf.matmul(tf.matmul(X, Sigma_p), X, transpose_b=True)
        else:
            return tf.matmul(tf.matmul(X, Sigma_p), X2, transpose_b=True)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        Sigma_p = tf.matmul(self.L_p, self.L_p, transpose_b=True)
        return tf.matrix_diag_part(tf.matmul(tf.matmul(X, Sigma_p), X, transpose_b=True))


# class ManifoldL2(kernels.Stationary):
#     def __init__(self, input_dim, variance=1.0, lengthscales=1.0, layer1=3, layer2=2,
#                  active_dims=None, ARD=None, name=None):
#         super().__init__(layer2, variance, lengthscales, active_dims, ARD, name)
#         np.random.seed(123)
#         # self.W1 = Parameter(value=np.ones(np.prod([layer1, input_dim]), dtype=settings.float_type), transform=None,
#         #                     prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
#         self.W1 = Parameter(value=self.initW(layer1, input_dim), transform=None,
#                             prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
#         self.b1 = Parameter(value=np.zeros(shape=[layer1, 1], dtype=settings.float_type), transform=None,
#                             prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
#         # self.W2 = Parameter(value=np.ones(np.prod([layer2, layer1]), dtype=settings.float_type), transform=None,
#         #                     prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
#         self.W2 = Parameter(value=self.initW(layer2, layer1), transform=None,
#                             prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
#         self.b2 = Parameter(value=np.zeros(shape=[layer2, 1], dtype=settings.float_type), transform=None,
#                             prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
#         self.input = input_dim
#         self.layer1 = layer1
#         self.layer2 = layer2
#
#     def initW(self, dim_in, dim_out):
#         return np.random.randn(dim_in, dim_out)     # * (2. / (dim_in + dim_out)) ** 0.5 crashes with step function test
#
#     def neural_network(self, X):
#         if X is None:
#             return X
#         # # X1 = X
#         # shapeW1 = [self.layer1, self.input]
#         # W1 = tf.reshape(self.W1, shape=shapeW1)
#         # shapeb1 = [self.layer1, 1]
#         # b1 = tf.reshape(self.b1, shape=shapeb1)
#         # shapeW2 = [self.layer2, self.layer1]
#         # W2 = tf.reshape(self.W2, shape=shapeW2)
#         # shapeb2 = [self.layer2, 1]
#         # b2 = tf.reshape(self.b2, shape=shapeb2)
#
#         X1 = tf.nn.sigmoid(tf.matmul(self.W1, X, transpose_b=True) + self.b1)
#         X2 = tf.transpose(tf.nn.sigmoid(tf.matmul(self.W2, X1) + self.b2))
#         return X2
#         # return X1
#
#     @params_as_tensors
#     def K(self, X, X2=None, presliced=False):
#         # if not presliced:
#         #     X, X2 = self._slice(X, X2)
#         X_out = self.neural_network(X)
#         X2_out = self.neural_network(X2)
#         return self.variance * tf.exp(-self.scaled_square_dist(X_out, X2_out) / 2)


class TreeCoregion(Kernel):
    def __init__(self, input_dim, output_dim, indices_tree, values, active_dims=None, name=None):
        '''

        :param input_dim:   dimension of the input to the kernel
        :param output_dim:  number of outputs of the multi-output GP (=self.output_dim, =D)
        :param indices_tree: list of tuples (or lists) containing ["diagonal" locations, "upper triangular" locations]  Not Symmetric yet!
        :param values:       array(D + Nij,): "D" diagonal terms and "Nij" off-diagonal upper triangular terms          Not Symmetric
        :param active_dims: dimensions to be selected
        :param name:
        '''

        """
        A Coregionalization kernel. The inputs to this kernel are _integers_
        (we cast them from floats as needed) which usually specify the
        *outputs* of a Coregionalization model.

        The parameters of this kernel, W, kappa, specify a positive-definite
        matrix B.

          B = tf.SparseTensor(indices_symm, tf.tanh(values_symm), dense_shape) .

        The kernel function is then an indexing of this matrix, so

          K(x, y) = B[x, y] .

        We refer to the size of B as "num_outputs x num_outputs", since this is
        the number of outputs in a coregionalization model. We refer to the
        number of columns on W as 'rank': it is the number of degrees of
        correlation between the outputs.

        NB. There is a symmetry between the elements of W, which creates a
        local minimum at W=0. To avoid this, it's recommended to initialize the
        optimization (or MCMC chain) using a random W.
        """
        assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(input_dim, active_dims, name=name)

        self.output_dim = output_dim
        # self.rank = rank
        # self.W = Parameter(np.zeros((self.output_dim, self.rank), dtype=settings.float_type))
        # self.kappa = Parameter(np.ones(self.output_dim, dtype=settings.float_type), transform=transforms.positive)

        self.dense_shape = [output_dim, output_dim]
        self.indices = indices_tree     #
        self.values = Parameter(values)   # needs to be initialized

    @params_as_tensors
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        X = tf.cast(X[:, 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[:, 0], tf.int32)
        # B = tf.matmul(self.W, self.W, transpose_b=True) + tf.matrix_diag(self.kappa)
        values_symm = tf.concat([self.values, self.values[self.output_dim:]], axis=0)           # concat([[00, 11, ..., (D-1)(D-1), ij1, ..., ijN],  [ji1, ..., jiN]])  # D = self.output_dim
        indices_symm = self.indices + self.symmetric_indices(self.indices[self.output_dim:])    #         [00, 11, ..., (D-1)(D-1), ij1, ..., ijN] + [ji1, ..., jiN]    # D = self.output_dim
        indices_symm = tf.constant(indices_symm, dtype=settings.int_type)
        B = tf.sparse_to_dense(sparse_indices=indices_symm, sparse_values=tf.tanh(values_symm),
                               output_shape=self.dense_shape, validate_indices=False) + \
            tf.eye(self.output_dim, dtype=settings.float_type) * self.output_dim                # needs to be symmetric!
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    @params_as_tensors
    def Kdiag(self, X):
        X, _ = self._slice(X, None)
        X = tf.cast(X[:, 0], tf.int32)
        # Bdiag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        Bdiag = tf.tanh(self.values[:self.output_dim]) + tf.ones_like(self.values[:self.output_dim]) * self.output_dim
        return tf.gather(Bdiag, X)

    def symmetric_indices(self, indices_in):
        indices_out = []
        for tuple_i in indices_in:
            indices_out.append([tuple_i[1], tuple_i[0]])
        return indices_out