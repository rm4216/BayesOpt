import tensorflow as tf
import numpy as np

from gpflow import settings
from gpflow import likelihoods
from gpflow import transforms
# from gpflow import kernels
from gpflow import features

from gpflow.params import Parameter, Parameterized
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.expectations import expectation
from gpflow.probability_distributions import DiagonalGaussian

from tfbo.models.gp_models import GPModel
from tfbo.models.GaBP import GaBP
# from gpflow.models.gpr import GPR
from gpflow.logdensities import multivariate_normal
from gpflow.params import DataHolder
from gpflow.params import Minibatch
from gpflow.conditionals import conditional, Kuu
from gpflow import kullback_leiblers
from gpflow.conditionals import base_conditional

logger = settings.logger()

class BC_GPLVM(GPModel):
    def __init__(self, X, proj_dim, Y, kern, M, nn, Z=None, X_prior_mean=None, X_prior_var=None):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.
        :param X_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_var: variance of latent positions (N x Q), for the initialisation of the latent space.
        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param kern: kernel specification, by default RBF
        :param M: number of inducing points
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
        random permutation of X_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_mean.
        :param X_prior_var: pripor variance used in KL term of bound. By default 1.
        """
        GPModel.__init__(self, X, Y, kern,
                         likelihood=likelihoods.Gaussian(),
                         mean_function=Zero())
        del self.X  # in GPLVM this is a Param
        self.proj_dim = proj_dim
        self.nn = nn
        self.X_mean = self.nn.forward(X)
        self.X_var = Parameter(0.1*np.ones_like(self.nn.np_forward(X)), dtype=settings.float_type, transform=transforms.positive)     # enforce positivity on outputs of the NN

        # self.X_mean = Parameter(X_mean)
        # diag_transform = transforms.DiagMatrix(X_var.shape[1])
        # self.X_var = Parameter(diag_transform.forward(transforms.positive.backward(X_var)) if X_var.ndim == 2 else X_var,
        #                    diag_transform)
        # assert X_var.ndim == 2
        # self.X_var = Parameter(X_var, transform=transforms.positive)

        self.num_data = X.shape[0]
        self.num_latent = proj_dim
        self.output_dim = Y.shape[1]

        # assert np.all(X_mean.shape == X_var.shape)
        assert X.shape[0] == Y.shape[0], 'X mean and Y must be same size.'
        # assert X_var.shape[0] == Y.shape[0], 'X var and Y must be same size.'

        # inducing points
        if Z is None:
            # By default we initialize by subset of initial latent points
            # Z = np.random.permutation(X.copy())[:M]
            Z = np.random.permutation(self.nn.np_forward(X).copy())[:M]

        self.feature = features.InducingPoints(Z)

        assert len(self.feature) == M
        # assert X_mean.shape[1] == self.num_latent

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = np.zeros((self.num_data, self.num_latent))
        if X_prior_var is None:
            X_prior_var = np.ones((self.num_data, self.num_latent))

        self.X_prior_mean = np.asarray(np.atleast_1d(X_prior_mean), dtype=settings.float_type)
        self.X_prior_var = np.asarray(np.atleast_1d(X_prior_var), dtype=settings.float_type)

        assert self.X_prior_mean.shape[0] == self.num_data
        assert self.X_prior_mean.shape[1] == self.num_latent
        assert self.X_prior_var.shape[0] == self.num_data
        assert self.X_prior_var.shape[1] == self.num_latent

        # self.kc = kc
        # self.input_dim = X.shape[1]
        # self.X_D = X

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        pX = DiagonalGaussian(self.X_mean, self.X_var)

        num_inducing = len(self.feature)
        psi0 = tf.reduce_sum(expectation(pX, self.kern))
        psi1 = expectation(pX, (self.kern, self.feature))
        psi2 = tf.reduce_sum(expectation(pX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        dX_var = self.X_var if len(self.X_var.get_shape()) == 2 else tf.matrix_diag_part(self.X_var)
        NQ = tf.cast(tf.size(self.X_mean), settings.float_type)
        D = tf.cast(tf.shape(self.Y)[1], settings.float_type)
        KL = -0.5 * tf.reduce_sum(tf.log(dX_var)) \
             + 0.5 * tf.reduce_sum(tf.log(self.X_prior_var)) \
             - 0.5 * NQ \
             + 0.5 * tf.reduce_sum((tf.square(self.X_mean - self.X_prior_mean) + dX_var) / self.X_prior_var)

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), settings.float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))
        bound -= KL


        # X_aug = tf.concat(
        #     [tf.concat([self.X_mean, tf.ones(shape=[self.X_mean.shape[0], 1], dtype=settings.float_type) * i], axis=1)
        #      for i in range(self.input_dim)], axis=0)
        # Y_aug = tf.concat([self.X_D[:, i] for i in range(self.input_dim)], axis=0)[:, None]  # need original X
        # K = self.kc.K(X_aug) + tf.eye(tf.shape(X_aug)[0], dtype=settings.float_type) * self.likelihood.variance
        # L = tf.cholesky(K)
        # m = self.mean_function(X_aug)
        # logpdf = multivariate_normal(Y_aug, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        # return tf.reduce_sum(logpdf)
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.
        :param Xnew: Point to predict at.
        """
        pX = DiagonalGaussian(self.X_mean, self.X_var)

        num_inducing = len(self.feature)
        psi1 = expectation(pX, (self.kern, self.feature))
        psi2 = tf.reduce_sum(expectation(pX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        Kus = self.feature.Kuf(self.kern, Xnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var


class NN(Parameterized):
    def __init__(self, dims, N, proj_dim, name=None):
        # super().__init__(self)
        Parameterized.__init__(self)
        np.random.seed(123)
        self.dims = dims
        self.input = dims[0]
        self.N = N
        self.Q = proj_dim
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            setattr(self, 'W_{}'.format(i), Parameter(value=self.initW(dim_out, dim_in), transform=None,
                                             prior=None, trainable=True, dtype=settings.float_type, fix_shape=True,
                                             name=name))
            setattr(self, 'b_{}'.format(i),
                    Parameter(value=np.zeros(shape=[dim_out, 1], dtype=settings.float_type), transform=None,
                              prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name))

        # setattr(self, 'W_{1}', Parameter(value=self.initW(layer1, input_dim), transform=None,
        #         prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name))
        # # self.W1 = Parameter(value=self.initW(layer1, input_dim), transform=None,
        # #                     prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
        # setattr(self, 'b_{1}', Parameter(value=np.zeros(shape=[layer1, 1], dtype=settings.float_type), transform=None,
        #         prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name))
        # # self.b1 = Parameter(value=np.zeros(shape=[layer1, 1], dtype=settings.float_type), transform=None,
        # #                     prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
        # setattr(self, 'W_{2}', Parameter(value=self.initW(layer2, layer1), transform=None,
        #         prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name))
        # # self.W2 = Parameter(value=self.initW(layer2, layer1), transform=None,
        # #                     prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)
        # setattr(self, 'b_{2}', Parameter(value=np.zeros(shape=[layer2, 1], dtype=settings.float_type), transform=None,
        #         prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name))
        # # self.b2 = Parameter(value=np.zeros(shape=[layer2, 1], dtype=settings.float_type), transform=None,
        # #                     prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name)


    def initW(self, dim_in, dim_out):
        return np.random.randn(dim_in, dim_out)

    @params_as_tensors
    def mean(self, X):
        if X is None:
            return X
        X = tf.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = tf.nn.sigmoid(tf.matmul(W, X) + b)
        X = tf.transpose(X)
        return tf.reshape(X[:, :self.Q], shape=[self.N, self.Q])     # first N*Q units reshaped
        # X1 = tf.nn.sigmoid(tf.matmul(self.W1, X, transpose_b=True) + self.b1)
        # X2 = tf.transpose(tf.nn.sigmoid(tf.matmul(self.W2, X1) + self.b2))
        # return tf.reshape(X2[:self.N*self.Q, :], shape=[self.N, self.Q])

    def np_mean(self, X):
        if X is None:
            return X
        def sigma(x):
            x = np.clip(x, a_min=-np.log(1e09), a_max=1e09)
            return np.divide(1., 1. + np.exp(-x))
        X = np.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = sigma(np.matmul(W.read_value(), X) + b.read_value())
        # X1 = sigma(np.matmul(self.W1.read_value(), np.transpose(X)) + self.b1.read_value())
        # X2 = np.transpose(sigma(np.matmul(self.W2.read_value(), X1) + self.b2.read_value()))
        assert X.shape[0] == 2 * self.Q
        X = np.transpose(X)
        return np.reshape(X[:, :self.Q], newshape=[self.N, self.Q])

    @params_as_tensors
    def var(self, X):
        if X is None:
            return X
        X = tf.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = tf.nn.sigmoid(
                tf.clip_by_value(tf.matmul(W, X) + b, clip_value_min=tf.constant(-np.log(1e09), dtype=settings.float_type),
                                 clip_value_max=tf.constant(1e09, dtype=settings.float_type)))
        X = tf.transpose(X)
        return tf.exp(tf.reshape(X[:, self.Q:], shape=[self.N, self.Q]))
        # X1 = tf.nn.sigmoid(tf.matmul(self.W1, X, transpose_b=True) + self.b1)
        # X2 = tf.transpose(tf.nn.sigmoid(tf.matmul(self.W2, X1) + self.b2))
        # return tf.exp(tf.reshape(X2[self.N*self.Q:, :], shape=[self.N, self.Q]))

    # def np_var(self, X):
    @params_as_tensors
    def forward(self, X):
        if X is None:
            return X
        X = tf.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = tf.nn.sigmoid(tf.matmul(W, X) + b)
        X = tf.transpose(X)
        return X

    def np_forward(self, X):
        if X is None:
            return X
        def sigma(x):
            x = np.clip(x, a_min=-np.log(1e09), a_max=1e09)
            return np.divide(1., 1. + np.exp(-x))
        X = np.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = sigma(np.matmul(W.read_value(), X) + b.read_value())
        # X1 = sigma(np.matmul(self.W1.read_value(), np.transpose(X)) + self.b1.read_value())
        # X2 = np.transpose(sigma(np.matmul(self.W2.read_value(), X1) + self.b2.read_value()))
        assert X.shape[0] == self.dims[-1]
        X = np.transpose(X)
        return X


class Ort_NN(Parameterized):
    def __init__(self, dims, N, proj_dim, name=None):
        # super().__init__(self)
        Parameterized.__init__(self)
        np.random.seed(123)
        self.dims = dims
        self.input = dims[0]
        self.N = N
        self.Q = proj_dim
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            setattr(self, 'W_{}'.format(i), Parameter(value=self.initW(dim_out, dim_in), transform=None,
                                             prior=None, trainable=True, dtype=settings.float_type, fix_shape=True,
                                             name=name))
            setattr(self, 'b_{}'.format(i),
                    Parameter(value=np.zeros(shape=[dim_out, 1], dtype=settings.float_type), transform=None,
                              prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name))


    def initW(self, dim_in, dim_out):
        return np.random.randn(dim_in, dim_out)

    # def np_var(self, X):
    @params_as_tensors
    def forward(self, X):
        if X is None:
            return X
        X = tf.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            if W.shape[0].value < W.shape[1].value:
                q, r = tf.linalg.qr(tf.transpose(W))
                q = tf.transpose(q)
            else:
                q, r = tf.linalg.qr(W)
            b = getattr(self, 'b_{}'.format(i))
            X = tf.nn.sigmoid(
                tf.clip_by_value(tf.matmul(q, X) + b, clip_value_min=tf.constant(-np.log(1e09), dtype=settings.float_type),
                                 clip_value_max=tf.constant(1e09, dtype=settings.float_type)))
        X = tf.transpose(X)
        return X

    def np_forward(self, X):
        if X is None:
            return X
        def sigma(x):
            x = np.clip(x, a_min=-np.log(1e09), a_max=1e09)
            return np.divide(1., 1. + np.exp(-x))
        X = np.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            if W.shape[0] < W.shape[1]:
                q, r = np.linalg.qr(np.transpose(W.read_value()), mode='reduced')
                q = np.transpose(q)
            else:
                q, r = np.linalg.qr(W.read_value())
            b = getattr(self, 'b_{}'.format(i))
            X = sigma(np.matmul(q, X) + b.read_value())
        # X1 = sigma(np.matmul(self.W1.read_value(), np.transpose(X)) + self.b1.read_value())
        # X2 = np.transpose(sigma(np.matmul(self.W2.read_value(), X1) + self.b2.read_value()))
        assert X.shape[0] == self.dims[-1]
        X = np.transpose(X)
        return X



class Linear_NN(Parameterized):
    def __init__(self, dims, N, proj_dim, name=None):
        # super().__init__(self)
        Parameterized.__init__(self)
        np.random.seed(123)
        self.dims = dims
        self.input = dims[0]
        self.N = N
        self.Q = proj_dim
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            setattr(self, 'W_{}'.format(i), Parameter(value=self.initW(dim_out, dim_in), transform=None,
                                             prior=None, trainable=True, dtype=settings.float_type, fix_shape=True,
                                             name=name))
            setattr(self, 'b_{}'.format(i),
                    Parameter(value=np.zeros(shape=[dim_out, 1], dtype=settings.float_type), transform=None,
                              prior=None, trainable=True, dtype=settings.float_type, fix_shape=True, name=name))


    def initW(self, dim_in, dim_out):
        return np.random.randn(dim_in, dim_out)

    @params_as_tensors
    def mean(self, X):
        if X is None:
            return X
        X = tf.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = tf.nn.sigmoid(tf.matmul(W, X) + b)
        X = tf.transpose(X)
        return tf.reshape(X[:, :self.Q], shape=[self.N, self.Q])     # first N*Q units reshaped
        # X1 = tf.nn.sigmoid(tf.matmul(self.W1, X, transpose_b=True) + self.b1)
        # X2 = tf.transpose(tf.nn.sigmoid(tf.matmul(self.W2, X1) + self.b2))
        # return tf.reshape(X2[:self.N*self.Q, :], shape=[self.N, self.Q])

    def np_mean(self, X):
        if X is None:
            return X
        def sigma(x):
            return np.divide(1., 1. + np.exp(-x))
        X = np.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = sigma(np.matmul(W.read_value(), X) + b.read_value())
        # X1 = sigma(np.matmul(self.W1.read_value(), np.transpose(X)) + self.b1.read_value())
        # X2 = np.transpose(sigma(np.matmul(self.W2.read_value(), X1) + self.b2.read_value()))
        assert X.shape[0] == 2 * self.Q
        X = np.transpose(X)
        return np.reshape(X[:, :self.Q], newshape=[self.N, self.Q])

    @params_as_tensors
    def var(self, X):
        if X is None:
            return X
        X = tf.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = tf.nn.sigmoid(tf.matmul(W, X) + b)
        X = tf.transpose(X)
        return tf.exp(tf.reshape(X[:, self.Q:], shape=[self.N, self.Q]))
        # X1 = tf.nn.sigmoid(tf.matmul(self.W1, X, transpose_b=True) + self.b1)
        # X2 = tf.transpose(tf.nn.sigmoid(tf.matmul(self.W2, X1) + self.b2))
        # return tf.exp(tf.reshape(X2[self.N*self.Q:, :], shape=[self.N, self.Q]))

    # def np_var(self, X):
    @params_as_tensors
    def forward(self, X):
        if X is None:
            return X
        X = tf.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = tf.matmul(W, X) + b
        X = tf.transpose(X)
        return X

    def np_forward(self, X):
        if X is None:
            return X
        def sigma(x):
            return np.divide(1., 1. + np.exp(-x))
        X = np.transpose(X)
        for i in range(len(self.dims) - 1):
            W = getattr(self, 'W_{}'.format(i))
            b = getattr(self, 'b_{}'.format(i))
            X = np.matmul(W.read_value(), X) + b.read_value()
        # X1 = sigma(np.matmul(self.W1.read_value(), np.transpose(X)) + self.b1.read_value())
        # X2 = np.transpose(sigma(np.matmul(self.W2.read_value(), X1) + self.b2.read_value()))
        assert X.shape[0] == self.dims[-1]
        X = np.transpose(X)
        return X


class MGPR(GPModel):
    """
    Manifold Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, nn, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.nn = nn

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        Xnn = self.nn.forward(self.X)
        K = self.kern.K(Xnn) + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * self.likelihood.variance
        jitter = self.stable_jitter(K)
        L = tf.cholesky(K + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * jitter)
        m = self.mean_function(Xnn)
        logpdf = multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        return tf.reduce_sum(logpdf)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Xnn = self.nn.forward(self.X)
        y = self.Y - self.mean_function(Xnn)
        Kmn = self.kern.K(Xnn, Xnew)
        Kmm_sigma = self.kern.K(Xnn) + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * self.likelihood.variance
        jitter = self.stable_jitter(Kmm_sigma)
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * jitter,
                                         Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_function(Xnew), f_var

    def stable_jitter(self, K):
        # determinant = tf.matrix_determinant(K)
        # det_const = tf.constant(0)
        # det_const = tf.Print(det_const, [det_const], "Determinant of K matrix: ")
        # det_const = tf.Print(det_const, ["Determinant of K matrix: ",  det_const])

        s = tf.linalg.svd(K, compute_uv=False, full_matrices=False)
        eig_min = tf.abs(tf.reduce_min(s))

        jitter_const = tf.constant(settings.jitter, dtype=settings.float_type, shape=[])
        zero_const = tf.constant(0., dtype=settings.float_type, shape=[])

        def add_jitter(eig_min, jitter_const):
            eig_min = tf.Print(eig_min, [eig_min])
            return tf.add(tf.abs(eig_min), jitter_const)
        true_fn = lambda : add_jitter(eig_min=eig_min, jitter_const=jitter_const)

        def skip_jitter(zero_const): return tf.add(zero_const, zero_const)
        false_fn = lambda : skip_jitter(zero_const=zero_const)

        # add jitter 1e-06 and the lowest eigenvalue of K only if the lowest eigenvalue is smaller than settings.jitter
        jitter = tf.cond(tf.less(eig_min, jitter_const),
                         true_fn,
                         false_fn)
        # gradient of absolute value could return nan
        jitter = tf.stop_gradient(jitter)
        return jitter


class Stable_GPR(GPModel):
    """
    Plane Gaussian Process Regression that only applies a jitter to the training covariance matrix based on its minimum eigen-values, that we will use for multi-output GP

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    """
    def __init__(self, X, Y, kern, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        jitter = self.stable_jitter(K)
        L = tf.cholesky(K + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * jitter)
        m = self.mean_function(self.X)
        logpdf = multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        return tf.reduce_sum(logpdf)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        y = self.Y - self.mean_function(self.X)
        Kmn = self.kern.K(self.X, Xnew)
        Kmm_sigma = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        jitter = self.stable_jitter(Kmm_sigma)
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * jitter,
                                         Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_function(Xnew), f_var

    def stable_jitter(self, K):
        # determinant = tf.matrix_determinant(K)
        # det_const = tf.constant(0)
        # det_const = tf.Print(det_const, [det_const], "Determinant of K matrix: ")
        # det_const = tf.Print(det_const, ["Determinant of K matrix: ",  det_const])

        s = tf.linalg.svd(K, compute_uv=False, full_matrices=False)
        eig_min = tf.abs(tf.reduce_min(s))

        jitter_const = tf.constant(settings.jitter, dtype=settings.float_type, shape=[])
        zero_const = tf.constant(0., dtype=settings.float_type, shape=[])

        def add_jitter(eig_min, jitter_const):
            eig_min = tf.Print(eig_min, [eig_min])
            return tf.add(tf.abs(eig_min), jitter_const)
        true_fn = lambda : add_jitter(eig_min=eig_min, jitter_const=jitter_const)

        def skip_jitter(zero_const): return tf.add(zero_const, zero_const)
        false_fn = lambda : skip_jitter(zero_const=zero_const)

        # add jitter 1e-06 and the lowest eigenvalue of K only if the lowest eigenvalue is smaller than settings.jitter
        jitter = tf.cond(tf.less(eig_min, jitter_const),
                         true_fn,
                         false_fn)
        # gradient of absolute value could return nan
        jitter = tf.stop_gradient(jitter)
        return jitter



class NN_MoGPR(GPModel):

    def __init__(self, X, Y, kern, nn, Mo_dim, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.nn = nn
        self.Mo_dim = Mo_dim
        self.input_dim = X.shape[1]
        self.decomposition = [list(np.arange(start=i*self.Mo_dim, stop=(i+1)*self.Mo_dim, step=1)) for i in
                              range(int(np.floor(self.input_dim/self.Mo_dim)))]
        self.k_indices = list(range(len(self.decomposition) + 1))     # all kernels for multiple-output and a regression kernel for Y

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        Xnn = self.nn.forward(self.X)
        Xnn_aug = tf.concat(
            [tf.concat([Xnn, tf.ones(shape=[tf.shape(Xnn)[0], 1], dtype=settings.float_type) * i], axis=1)
             for i in range(self.Mo_dim)], axis=0)
        Y_aug = tf.stack([tf.concat([self.X[:, i][:, None] for i in decomp_i], axis=0) for decomp_i in self.decomposition], axis=0)
        K = self.kern.K(Xnn_aug, i=self.k_indices[:-1]) + tf.eye(tf.shape(Xnn_aug)[0], dtype=settings.float_type)[None] * self.likelihood.variance
        # jitter = self.stable_jitter_vec(K)
        # jitter_mat = tf.multiply(tf.tile(tf.eye(tf.shape(Xnn_aug)[0], dtype=settings.float_type)[None], [tf.shape(K)[0], 1, 1]), jitter[:, None, None])
        # L = tf.cholesky(K + jitter_mat)
        L = tf.cholesky(K)
        m = self.mean_function(Xnn_aug)[None]

        Ky = self.kern.K(Xnn, i=self.k_indices[-1]) + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * self.likelihood.variance
        jittery = self.stable_jitter(Ky)
        Ly = tf.cholesky(Ky + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * jittery)
        my = self.mean_function(Xnn)
        logpdfy = multivariate_normal(self.Y, my, Ly)  # (R,) log-likelihoods for each independent dimension of Y

        d = Y_aug - m
        alpha = tf.matrix_triangular_solve(L, d, lower=True)
        # num_dims = tf.cast(tf.shape(d)[0], L.dtype) * tf.cast(tf.shape(d)[1], L.dtype)
        num_dims = tf.cast(tf.shape(d)[1], L.dtype)
        logpdf = - 0.5 * tf.reduce_sum(tf.square(alpha), 1)
        logpdf -= 0.5 * num_dims * np.log(2 * np.pi)
        logpdf -= tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=1, keepdims=True)
        # return logpdf, logpdfy[:, None]
        # return jitter[:, None, None], jitter_mat
        return tf.reduce_sum(logpdf) + logpdfy

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):

        Xnn = self.nn.forward(self.X)
        y = self.Y - self.mean_function(Xnn)
        Kmn = self.kern.K(Xnn, Xnew, i=self.k_indices[-1])
        Kmm_sigma = self.kern.K(Xnn, i=self.k_indices[-1]) + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * self.likelihood.variance
        jitter = self.stable_jitter(Kmm_sigma)
        Knn = self.kern.K(Xnew, i=self.k_indices[-1]) if full_cov else self.kern.Kdiag(Xnew, i=self.k_indices[-1])
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * jitter,
                                         Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_function(Xnew), f_var

    @params_as_tensors
    def _build_predict_x(self, Xnew, full_cov=False):
        Xnn = self.nn.forward(self.X)
        Xnn_aug = tf.concat(
            [tf.concat([Xnn, tf.ones(shape=[tf.shape(Xnn)[0], 1], dtype=settings.float_type) * i], axis=1)
             for i in range(self.Mo_dim)], axis=0)
        Y_aug = tf.stack([tf.concat([self.X[:, i][:, None] for i in decomp_i], axis=0) for decomp_i in self.decomposition], axis=0)
        K = self.kern.K(Xnn_aug, i=self.k_indices[:-1]) + tf.eye(tf.shape(Xnn_aug)[0], dtype=settings.float_type)[None] * self.likelihood.variance
        jitter = self.stable_jitter_vec(K)
        jitter_mat = tf.multiply(tf.tile(tf.eye(tf.shape(Xnn_aug)[0], dtype=settings.float_type)[None], [tf.shape(K)[0], 1, 1]), jitter[:, None, None])
        L = tf.cholesky(K + jitter_mat)
        m = self.mean_function(Xnn_aug)[None]
        d = Y_aug - m

        # Compute the projection matrix A
        Xnew_aug = tf.concat(
            [tf.concat([Xnew, tf.ones(shape=[tf.shape(Xnew)[0], 1], dtype=settings.float_type) * i], axis=1)
             for i in range(self.Mo_dim)], axis=0)
        Kmn = self.kern.K(Xnn_aug, Xnew_aug, i=self.k_indices[:-1])
        A = tf.matrix_triangular_solve(L, Kmn, lower=True)

        Knn = self.kern.K(Xnew_aug, i=self.k_indices[:-1]) if full_cov else self.kern.Kdiag(Xnew_aug, i=self.k_indices[:-1])
        # Knn = tf.matrix_diag_part(self.kern.K(Xnew_aug, i=self.k_indices[:-1]))[:, :, None]
        # compute the covariance due to the conditioning
        if full_cov:
            fvar = Knn - tf.matmul(A, A, transpose_a=True)
        else:
            fvar = Knn - tf.reduce_sum(tf.square(A), 1)

        # another backsubstitution in the unwhitened case
        A = tf.matrix_triangular_solve(tf.transpose(L, perm=[0, 2, 1]), A, lower=False)
        # construct the conditional mean
        fmean = tf.matmul(A, d, transpose_a=True)
        # if not full_cov:
        #     fvar = tf.transpose(fvar)  # N x R
        # fmean_m = fmean + self.mean_function(Xnew_aug)[None]
        # fmean_reshape = tf.reshape(tf.transpose(fmean_m, perm=[0, 2, 1]), [])
        return fmean + self.mean_function(Xnew_aug)[None], fvar


    def stable_jitter_vec(self, K):

        s = tf.linalg.svd(K, compute_uv=False, full_matrices=False)
        eig_min = tf.abs(tf.reduce_min(s, axis=-1))

        jitter_const = tf.constant(settings.jitter, dtype=settings.float_type, shape=[])
        # zero_const = tf.constant(0., dtype=settings.float_type, shape=[])
        zero_const = tf.zeros_like(eig_min, dtype=settings.float_type)

        def add_jitter(eig_min, jitter_const):
            eig_min = tf.Print(eig_min, [eig_min])
            return tf.add(tf.abs(eig_min), jitter_const)
        true_fn = lambda : add_jitter(eig_min=eig_min, jitter_const=jitter_const)
        # false_fn = lambda : add_jitter(eig_min=eig_min, jitter_const=jitter_const)

        def skip_jitter(zero_const): return tf.add(zero_const, zero_const)
        false_fn = lambda : skip_jitter(zero_const=zero_const)
        # true_fn = lambda : skip_jitter(zero_const=zero_const)

        # add jitter 1e-06 and the lowest eigenvalue of K only if the lowest eigenvalue is smaller than settings.jitter
        jitter = tf.cond(tf.reduce_any(tf.less(eig_min, jitter_const)),
                         true_fn,
                         false_fn)
        # gradient of absolute value could return nan
        jitter = tf.stop_gradient(jitter)
        return jitter

    def stable_jitter(self, K):
        # determinant = tf.matrix_determinant(K)
        # det_const = tf.constant(0)
        # det_const = tf.Print(det_const, [det_const], "Determinant of K matrix: ")
        # det_const = tf.Print(det_const, ["Determinant of K matrix: ",  det_const])

        s = tf.linalg.svd(K, compute_uv=False, full_matrices=False)
        eig_min = tf.abs(tf.reduce_min(s))

        jitter_const = tf.constant(settings.jitter, dtype=settings.float_type, shape=[])
        zero_const = tf.constant(0., dtype=settings.float_type, shape=[])

        def add_jitter(eig_min, jitter_const):
            eig_min = tf.Print(eig_min, [eig_min])
            return tf.add(tf.abs(eig_min), jitter_const)
        true_fn = lambda : add_jitter(eig_min=eig_min, jitter_const=jitter_const)

        def skip_jitter(zero_const): return tf.add(zero_const, zero_const)
        false_fn = lambda : skip_jitter(zero_const=zero_const)

        # add jitter 1e-06 and the lowest eigenvalue of K only if the lowest eigenvalue is smaller than settings.jitter
        jitter = tf.cond(tf.less(eig_min, jitter_const),
                         true_fn,
                         false_fn)
        # gradient of absolute value could return nan
        jitter = tf.stop_gradient(jitter)
        return jitter


class FastNN_MoGPR(GPModel):

    def __init__(self, X, Y, kern, nn, Mo_dim, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.nn = nn
        self.Mo_dim = Mo_dim
        self.input_dim = X.shape[1]
        self.decomposition = [list(np.arange(start=i*self.Mo_dim, stop=(i+1)*self.Mo_dim, step=1)) for i in
                              range(int(np.floor(self.input_dim/self.Mo_dim)))]
        self.k_indices = list(range(len(self.decomposition) + 1))     # all kernels for multiple-output and a regression kernel for Y
        self.evaluate_B_matrix = int(1)
        self.evaluate_base_kernel = int(0)
        self.evaluate_response_surface_k = int(2)

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        Xnn = self.nn.forward(self.X)

        K = self.kern.K(Xnn, i=self.k_indices[:-1], base_coregion=self.evaluate_base_kernel)    # (20, N, N) = Kmatrix

        XB = tf.concat([tf.concat([tf.zeros(shape=[1, tf.shape(Xnn)[1]], dtype=settings.float_type),
                                   tf.ones(shape=[1, 1], dtype=settings.float_type) * i], axis=1) for i in
                        range(self.Mo_dim)], axis=0)                                            # (self.Mo_dim=3, d) = XBmatrix; d=dimensionality feature space, i.e. d != self.Mo_dim

        B = self.kern.K(XB, i=self.k_indices[:-1], base_coregion=self.evaluate_B_matrix)        # (20, self.Mo_dim, self.Mo_dim)

        N = tf.shape(self.X)[0]
        y_vec = tf.reshape(tf.transpose(self.X), shape=[len(self.decomposition), self.Mo_dim * N, 1])
        m = tf.stack([self.mean_function(y_vec[i, :, :]) for i in range(len(self.decomposition))], axis=0)
        X_vec = y_vec - m           # check shapes are the same!!

        # # Eigen-decomposition
        l_k, q_k = tf.linalg.eigh(K)
        l_b, q_b = tf.linalg.eigh(B)


        QbQkX_vec = self.mat_vec_mul(tf.transpose(q_b, perm=[0, 2, 1]), tf.transpose(q_k, perm=[0, 2, 1]), X_vec)
        kron_diag = tf.stack(
            [tf.concat([l_k[comp_i, :, None] * l_b[comp_i, i] for i in range(self.Mo_dim)], axis=0) for comp_i in
             range(len(self.decomposition))], axis=0)

        Inv_vec = QbQkX_vec / (kron_diag + self.likelihood.variance)

        alpha = self.mat_vec_mul(q_b, q_k, Inv_vec)

        logpdf = tf.reduce_sum(-0.5 * tf.matmul(X_vec, alpha, transpose_a=True), axis=0)
        # num_dims = tf.cast(tf.shape(d)[1], L.dtype)
        num_dims = tf.cast(tf.shape(X_vec)[1], dtype=X_vec.dtype)
        logpdf -= 0.5 * num_dims * np.log(2 * np.pi) * tf.constant(len(self.decomposition), dtype=X_vec.dtype)
        # logpdf -= 0.5 * tf.cast(tf.shape(X_vec)[1], dtype=settings.float_type) * np.log(2 * np.pi)
        logpdf -= 0.5 * tf.reduce_sum(tf.log(kron_diag + self.likelihood.variance))

        Ky = self.kern.K(Xnn, i=self.k_indices[-1], base_coregion=self.evaluate_response_surface_k) + tf.eye(
            tf.shape(Xnn)[0], dtype=settings.float_type) * self.likelihood.variance
        jittery = self.stable_jitter(Ky)
        Ly = tf.cholesky(Ky + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * jittery)
        my = self.mean_function(Xnn)
        logpdfy = multivariate_normal(self.Y, my, Ly)  # (R,) log-likelihoods for each independent dimension of Y

        return (logpdf[0, :] / self.input_dim) + logpdfy    # , tf.cast(tf.shape(X_vec)[1], dtype=settings.float_type)[None]

    def mat_vec_mul(self, B, K, X_vec):
        '''
        Efficient matrix-vector multiplication where
        the matrix is Kronecker product $B \cross K$,
        and the vector is $X_vec$
        :param B: First term Kronecker product
        :param K: Second term Kronecker product
        :param X_vec: vector of shape (Gb*Gk, 1)
        :return: matmul(Kron(B, K), X_vec)
        '''

        Gb = tf.shape(B)[1]
        Gk = tf.shape(K)[1]

        X_Gk = tf.reshape(X_vec, shape=[len(self.decomposition), Gb, Gk])
        Z = tf.matmul(X_Gk, K, transpose_b=True)        # transposition?
        Z_vec = tf.reshape(tf.transpose(Z, perm=[0, 2, 1]), shape=[len(self.decomposition), Gb * Gk, 1])

        Z_Gb = tf.reshape(Z_vec, shape=[len(self.decomposition), Gk, Gb])
        M = tf.matmul(Z_Gb, B, transpose_b=True)        # transposition?
        x_out = tf.reshape(tf.transpose(M, perm=[0, 2, 1]), shape=[len(self.decomposition), Gb * Gk, 1])
        return x_out


    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):

        Xnn = self.nn.forward(self.X)
        y = self.Y - self.mean_function(Xnn)
        Kmn = self.kern.K(Xnn, Xnew, i=self.k_indices[-1], base_coregion=self.evaluate_response_surface_k)

        Kmm_sigma = self.kern.K(Xnn, i=self.k_indices[-1], base_coregion=self.evaluate_response_surface_k) + tf.eye(
            tf.shape(Xnn)[0], dtype=settings.float_type) * self.likelihood.variance
        jitter = self.stable_jitter(Kmm_sigma)
        Knn = self.kern.K(Xnew, i=self.k_indices[-1],
                          base_coregion=self.evaluate_response_surface_k) if full_cov else self.kern.Kdiag(Xnew, i=
        self.k_indices[-1], base_coregion=self.evaluate_response_surface_k)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * jitter,
                                         Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N

        return f_mean + self.mean_function(Xnew), f_var

    @params_as_tensors
    def _build_predict_x(self, Xnew, full_cov=False):
        Xnn = self.nn.forward(self.X)

        K = self.kern.K(Xnn, i=self.k_indices[:-1], base_coregion=self.evaluate_base_kernel)    # (20, N, N) = Kmatrix

        XB = tf.concat([tf.concat([tf.zeros(shape=[1, tf.shape(Xnn)[1]], dtype=settings.float_type),
                                   tf.ones(shape=[1, 1], dtype=settings.float_type) * i], axis=1) for i in
                        range(self.Mo_dim)], axis=0)                                            # (self.Mo_dim=3, d) = XBmatrix; d=dimensionality feature space, i.e. d != self.Mo_dim

        B = self.kern.K(XB, i=self.k_indices[:-1], base_coregion=self.evaluate_B_matrix)        # (20, self.Mo_dim, self.Mo_dim)


        N = tf.shape(self.X)[0]
        y_vec = tf.reshape(tf.transpose(self.X), shape=[len(self.decomposition), self.Mo_dim * N, 1])
        m = tf.stack([self.mean_function(y_vec[i, :, :]) for i in range(len(self.decomposition))], axis=0)
        X_vec = y_vec - m           # check shapes are the same!!


        # # Eigen-decomposition
        l_k, q_k = tf.linalg.eigh(K)
        l_b, q_b = tf.linalg.eigh(B)


        QbQkX_vec = self.mat_vec_mul(tf.transpose(q_b, perm=[0, 2, 1]), tf.transpose(q_k, perm=[0, 2, 1]), X_vec)
        kron_diag = tf.stack(
            [tf.concat([l_k[comp_i, :, None] * l_b[comp_i, i] for i in range(self.Mo_dim)], axis=0) for comp_i in
             range(len(self.decomposition))], axis=0)

        Inv_vec = QbQkX_vec / (kron_diag + self.likelihood.variance)

        alpha = self.mat_vec_mul(q_b, q_k, Inv_vec)

        Kmn = self.kern.K(Xnew, Xnn, i=self.k_indices[:-1], base_coregion=self.evaluate_base_kernel)
        f_mean = self.mat_vec_mul(B, Kmn, alpha)


        # posterior covariance matrix !! For a single test point !!
        Kmm = self.kern.K(Xnew, i=self.k_indices[:-1], base_coregion=self.evaluate_base_kernel)     # (20, 1, 1)
        BLin = tf.linalg.LinearOperatorFullMatrix(B)
        KmmLin = tf.linalg.LinearOperatorFullMatrix(Kmm)
        kss = tf.linalg.LinearOperatorKronecker([BLin, KmmLin]).to_dense()                          # (20, self.Mo_dim, self.Mo_dim)

        KnmLin = tf.linalg.LinearOperatorFullMatrix(tf.transpose(Kmn, perm=[0, 2, 1]))
        Ks = tf.linalg.LinearOperatorKronecker([BLin, KnmLin]).to_dense()                           # (20, self.Mo_dim*N, self.Mo_dim)
        # return f_mean + self.mean_function(tf.tile(Xnew, multiples=[self.Mo_dim, 1]))[
        #     None], Kmm, B, kss, Kmn, Ks
        mat_cov = tf.concat([self.mat_vec_mul(B, Kmn, self.mat_vec_mul(q_b, q_k, self.mat_vec_mul(
            tf.transpose(q_b, perm=[0, 2, 1]), tf.transpose(q_k, perm=[0, 2, 1]), Ks[:, :, i][:, :, None]) / (
                                                                                   kron_diag + self.likelihood.variance)))
                             for i in range(self.Mo_dim)], axis=2)                                  # (20, self.Mo_dim, self.Mo_dim)
        post_cov = kss - mat_cov
        # mat0 = self.mat_vec_mul(tf.transpose(q_b, perm=[0, 2, 1]), tf.transpose(q_k, perm=[0, 2, 1]), Ks[:, :, 0][:, :, None])
        # mat1 = mat0 / (kron_diag + self.likelihood.variance)
        # mat2 = self.mat_vec_mul(q_b, q_k, mat1)
        # mat3 = self.mat_vec_mul(B, Kmn, mat2)
        # return f_mean + self.mean_function(tf.tile(Xnew, multiples=[self.Mo_dim, 1]))[None], Kmm, B, kss, Kmn, Ks, mat_cov, post_cov

        # Xnn_aug = tf.concat(
        #     [tf.concat([Xnn, tf.ones(shape=[tf.shape(Xnn)[0], 1], dtype=settings.float_type) * i], axis=1)
        #      for i in range(self.Mo_dim)], axis=0)
        # Y_aug = tf.stack([tf.concat([self.X[:, i][:, None] for i in decomp_i], axis=0) for decomp_i in self.decomposition], axis=0)
        # K = self.kern.K(Xnn_aug, i=self.k_indices[:-1]) + tf.eye(tf.shape(Xnn_aug)[0], dtype=settings.float_type)[None] * self.likelihood.variance
        # jitter = self.stable_jitter_vec(K)
        # jitter_mat = tf.multiply(tf.tile(tf.eye(tf.shape(Xnn_aug)[0], dtype=settings.float_type)[None], [tf.shape(K)[0], 1, 1]), jitter[:, None, None])
        # L = tf.cholesky(K + jitter_mat)
        # m = self.mean_function(Xnn_aug)[None]
        # d = Y_aug - m
        #
        # # Compute the projection matrix A
        # Xnew_aug = tf.concat(
        #     [tf.concat([Xnew, tf.ones(shape=[tf.shape(Xnew)[0], 1], dtype=settings.float_type) * i], axis=1)
        #      for i in range(self.Mo_dim)], axis=0)
        # Kmn = self.kern.K(Xnn_aug, Xnew_aug, i=self.k_indices[:-1])
        # A = tf.matrix_triangular_solve(L, Kmn, lower=True)
        #
        # Knn = self.kern.K(Xnew_aug, i=self.k_indices[:-1]) if full_cov else self.kern.Kdiag(Xnew_aug, i=self.k_indices[:-1])
        # # Knn = tf.matrix_diag_part(self.kern.K(Xnew_aug, i=self.k_indices[:-1]))[:, :, None]
        # # compute the covariance due to the conditioning
        # if full_cov:
        #     fvar = Knn - tf.matmul(A, A, transpose_a=True)
        # else:
        #     fvar = Knn - tf.reduce_sum(tf.square(A), 1)
        #
        # # another backsubstitution in the unwhitened case
        # A = tf.matrix_triangular_solve(tf.transpose(L, perm=[0, 2, 1]), A, lower=False)
        # # construct the conditional mean
        # fmean = tf.matmul(A, d, transpose_a=True)
        # # if not full_cov:
        # #     fvar = tf.transpose(fvar)  # N x R
        # # fmean_m = fmean + self.mean_function(Xnew_aug)[None]
        # # fmean_reshape = tf.reshape(tf.transpose(fmean_m, perm=[0, 2, 1]), [])





        # Xnn = self.nn.forward(self.X)
        #
        # K = self.kern.K(Xnn, i=self.kk)
        # XB = tf.concat([tf.concat([tf.zeros(shape=[1, tf.shape(Xnn)[1]], dtype=settings.float_type),
        #                            tf.ones(shape=[1, 1], dtype=settings.float_type) * i], axis=1) for i in
        #                 range(self.Mo_dim)], axis=0)
        # B = self.kern.K(XB, i=self.kb)
        #
        # N = tf.shape(self.X)[0]
        # y_vec = tf.reshape(tf.transpose(self.X), shape=[self.Mo_dim * N, 1])
        # m = self.mean_function(y_vec)
        # X_vec = y_vec - m
        #
        # l_k, q_k = tf.linalg.eigh(K)
        # l_b, q_b = tf.linalg.eigh(B)
        #
        # QbQkX_vec = self.mat_vec_mul(tf.transpose(q_b), tf.transpose(q_k), X_vec)
        # kron_diag = tf.concat([l_k[:, None] * l_b[i] for i in range(self.Mo_dim)], axis=0)
        #
        # Inv_vec = QbQkX_vec / (kron_diag + self.likelihood.variance)
        #
        # alpha = self.mat_vec_mul(q_b, q_k, Inv_vec)
        #
        # Kmn = self.kern.K(Xnew, Xnn, i=self.kk)
        # f_mean = self.mat_vec_mul(B, Kmn, alpha)

        # return f_mean + self.mean_function(tf.tile(Xnew, multiples=[self.Mo_dim, 1]))

        return f_mean + self.mean_function(tf.tile(Xnew, multiples=[self.Mo_dim, 1]))[None], post_cov    # self.mean_function(Xnew_aug)[None], fvar


    def stable_jitter_vec(self, K):

        s = tf.linalg.svd(K, compute_uv=False, full_matrices=False)
        eig_min = tf.abs(tf.reduce_min(s, axis=-1))

        jitter_const = tf.constant(settings.jitter, dtype=settings.float_type, shape=[])
        # zero_const = tf.constant(0., dtype=settings.float_type, shape=[])
        zero_const = tf.zeros_like(eig_min, dtype=settings.float_type)

        def add_jitter(eig_min, jitter_const):
            eig_min = tf.Print(eig_min, [eig_min])
            return tf.add(tf.abs(eig_min), jitter_const)
        true_fn = lambda : add_jitter(eig_min=eig_min, jitter_const=jitter_const)
        # false_fn = lambda : add_jitter(eig_min=eig_min, jitter_const=jitter_const)

        def skip_jitter(zero_const): return tf.add(zero_const, zero_const)
        false_fn = lambda : skip_jitter(zero_const=zero_const)
        # true_fn = lambda : skip_jitter(zero_const=zero_const)

        # add jitter 1e-06 and the lowest eigenvalue of K only if the lowest eigenvalue is smaller than settings.jitter
        jitter = tf.cond(tf.reduce_any(tf.less(eig_min, jitter_const)),
                         true_fn,
                         false_fn)
        # gradient of absolute value could return nan
        jitter = tf.stop_gradient(jitter)
        return jitter

    def stable_jitter(self, K):
        # determinant = tf.matrix_determinant(K)
        # det_const = tf.constant(0)
        # det_const = tf.Print(det_const, [det_const], "Determinant of K matrix: ")
        # det_const = tf.Print(det_const, ["Determinant of K matrix: ",  det_const])

        s = tf.linalg.svd(K, compute_uv=False, full_matrices=False)
        eig_min = tf.abs(tf.reduce_min(s))

        jitter_const = tf.constant(settings.jitter, dtype=settings.float_type, shape=[])
        zero_const = tf.constant(0., dtype=settings.float_type, shape=[])

        def add_jitter(eig_min, jitter_const):
            eig_min = tf.Print(eig_min, [eig_min])
            return tf.add(tf.abs(eig_min), jitter_const)
        true_fn = lambda : add_jitter(eig_min=eig_min, jitter_const=jitter_const)

        def skip_jitter(zero_const): return tf.add(zero_const, zero_const)
        false_fn = lambda : skip_jitter(zero_const=zero_const)

        # add jitter 1e-06 and the lowest eigenvalue of K only if the lowest eigenvalue is smaller than settings.jitter
        jitter = tf.cond(tf.less(eig_min, jitter_const),
                         true_fn,
                         false_fn)
        # gradient of absolute value could return nan
        jitter = tf.stop_gradient(jitter)
        return jitter


class NN_FullMoGP(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, nn, Mo_dim, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.nn = nn
        self.Mo_dim = Mo_dim
        # Assuming kern in input is "Multiple_k" kernel with:
        # kern.K(, i=0) = gpflow.kernels.Coregion           Coregionalization kernel MOGP
        # kern.K(, i=1) = gpflow.kernels.Matern52/RBF/etc.  Standard kernel MOGP
        # kern.K(, i=2) = gpflow.kernels.Matern52/RBF/etc.  Standard kernel Manifold GP
        self.kb = int(0)    # index Coregion kernel
        self.kk = int(1)    # index MOGP kernel
        self.km = int(2)    # index Manifold GP kernel


    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """

        Xnn = self.nn.forward(self.X)
        # Xnn = tf.Print(Xnn, [Xnn])

        # Marginal likelihood from MOGP
        # K = self.kern.kernels[0].K(self.X)
        K = self.kern.K(Xnn, i=self.kk)
        XB = tf.concat([tf.concat([tf.zeros(shape=[1, tf.shape(Xnn)[1]], dtype=settings.float_type),
                                   tf.ones(shape=[1, 1], dtype=settings.float_type) * i], axis=1) for i in
                        range(self.Mo_dim)], axis=0)
        # B = self.kern.kernels[1].K(XB)
        B = self.kern.K(XB, i=self.kb)
        # return K, B

        N = tf.shape(self.X)[0]
        y_vec = tf.reshape(tf.transpose(self.X), shape=[self.Mo_dim * N, 1])
        m = self.mean_function(y_vec)
        X_vec = y_vec - m

        # # Efficient matrix-vector multiplication
        # # X_Gk = tf.reshape(X_vec, shape=[self.Mo_dim, N])
        # # Z = tf.matmul(X_Gk, K, transpose_b=True)
        # # Z_vec = tf.reshape(tf.transpose(Z), shape=[self.Mo_dim * N, 1])
        # #
        # # Z_Gb = tf.reshape(Z_vec, shape=[N, self.Mo_dim])
        # # M = tf.matmul(Z_Gb, B, transpose_b=True)
        # # x_out = tf.reshape(tf.transpose(M), shape=[self.Mo_dim * N, 1])
        # x_out = self.mat_vec_mul(B, K, X_vec)
        # return x_out


        # # Eigen-decomposition
        # K = tf.Print(K, [K])
        l_k, q_k = tf.linalg.eigh(K)
        l_b, q_b = tf.linalg.eigh(B)
        # K_new = tf.matmul(tf.matmul(q_k, tf.matrix_diag(l_k)), q_k, transpose_b=True)
        # B_new = tf.matmul(tf.matmul(q_b, tf.matrix_diag(l_b)), q_b, transpose_b=True)
        # return K_new, B_new, K, B

        # max_diff = tf.reduce_max(tf.abs(A - B))
        # test_ab = tf.assert_less(max_diff, tf.cast(1e-12, max_diff.dtype),
        #                            name='test_ab', data=[A, B])
        with tf.control_dependencies([tf.check_numerics(l_k, 'l_k e-values')]):     # needs to go through the checking to continue
            l_k = tf.identity(l_k)

        with tf.control_dependencies([tf.check_numerics(l_b, 'l_b e-values')]):     # needs to go through the checking to continue
            l_b = tf.identity(l_b)

        with tf.control_dependencies([tf.check_numerics(q_k, 'q_k e-vects')]):     # needs to go through the checking to continue
            q_k = tf.identity(q_k)

        with tf.control_dependencies([tf.check_numerics(q_b, 'q_b e-vects')]):     # needs to go through the checking to continue
            q_b = tf.identity(q_b)

        QbQkX_vec = self.mat_vec_mul(tf.transpose(q_b), tf.transpose(q_k), X_vec)
        kron_diag = tf.concat([l_k[:, None] * l_b[i] for i in range(self.Mo_dim)], axis=0)

        # Lk = tf.cholesky(K)
        # Lb = tf.cholesky(B)
        # Lk_inv = tf.matrix_triangular_solve(Lk, tf.eye(tf.shape(Lk)[0], dtype=settings.float_type), lower=True)
        # Lb_inv = tf.matrix_triangular_solve(Lb, tf.eye(tf.shape(Lb)[0], dtype=settings.float_type), lower=True)
        # K_inv = tf.matmul(Lk_inv, Lk_inv, transpose_a=True)
        # B_inv = tf.matmul(Lb_inv, Lb_inv, transpose_a=True)
        # t_k, qt_k = tf.linalg.eigh(K_inv)
        # t_b, qt_b = tf.linalg.eigh(B_inv)
        #
        # kron_t_diag = tf.concat([t_k[:, None] * t_b[i] for i in range(self.Mo_dim)], axis=0)
        # t_factor = kron_t_diag * (1. / 1. + kron_t_diag * self.likelihood.variance)
        #
        # Inv_vec = tf.multiply(QbQkX_vec, t_factor)


        # min_diag = tf.reduce_min(kron_diag)
        # min_diag = tf.Print(min_diag, [min_diag])
        # with tf.control_dependencies([tf.check_numerics(min_diag, 'min diag')]):     # needs to go through the checking to continue
        #     min_diag = tf.identity(min_diag)
        # with tf.control_dependencies([tf.Print(min_diag, [min_diag])]):     # needs to go through the checking to continue
        #     min_diag = tf.identity(min_diag)





        # kron_diag_nn = tf.where(tf.greater(kron_diag, 0), kron_diag, tf.zeros_like(kron_diag))
        # Inv_vec = QbQkX_vec / (kron_diag_nn + 1e-06 + self.likelihood.variance)
        Inv_vec = QbQkX_vec / (kron_diag + self.likelihood.variance)
        # Inv_vec = QbQkX_vec / (kron_diag + 1e-06 + self.likelihood.variance)
        # Inv_vec = QbQkX_vec / (kron_diag + min_diag + self.likelihood.variance)

        with tf.control_dependencies([tf.check_numerics(Inv_vec, 'inv not nan')]):     # needs to go through the checking to continue
            Inv_vec = tf.identity(Inv_vec)

        alpha = self.mat_vec_mul(q_b, q_k, Inv_vec)
        # return QbQkX_vec, Inv_vec, alpha              # tested

        logpdf = -0.5 * tf.matmul(X_vec, alpha, transpose_a=True)
        logpdf -= 0.5 * tf.cast(tf.shape(X_vec)[0], dtype=settings.float_type) * np.log(2 * np.pi)
        # logpdf -= 0.5 * tf.reduce_sum(tf.log(kron_diag_nn + self.likelihood.variance))
        logpdf -= 0.5 * tf.reduce_sum(tf.log(kron_diag + self.likelihood.variance))



        # X_aug = tf.concat(
        #     [tf.concat([Xnn, tf.ones(shape=[tf.shape(Xnn)[0], 1], dtype=settings.float_type) * i], axis=1)
        #      for i in range(self.Mo_dim)], axis=0)
        # Y_aug = tf.concat([self.X[:, i][:, None] for i in range(self.Mo_dim)], axis=0)
        # m_yaug = self.mean_function(Y_aug)
        # K_mat = self.kern.K(X_aug) + tf.eye(tf.shape(X_aug)[0], dtype=settings.float_type) * self.likelihood.variance
        # L_mat = tf.cholesky(K_mat)
        # log_density = multivariate_normal(Y_aug, m_yaug, L_mat)
        #
        # # return X_aug
        # # return K_mat, L_mat
        # # return logpdf, tf.reduce_sum(log_density)[None, None]
        # return logpdf[0][0], tf.reduce_sum(log_density)


        # Marginal likelihood contribution of the Manifold GP
        Km = self.kern.K(Xnn, i=self.km) + tf.eye(tf.shape(Xnn)[0],
                                                  dtype=settings.float_type) * self.likelihood.variance
        # Km = tf.Print(Km, [Km])
        Lm = tf.cholesky(Km)
        mm = self.mean_function(Xnn)
        logpdfm = multivariate_normal(self.Y, mm, Lm)
        # return logpdf

        return (logpdf[0][0] / self.Mo_dim) + tf.reduce_sum(logpdfm)


    @params_as_tensors
    def _test_likelihood(self):

        Xnn = self.nn.forward(self.X)

        K = self.kern.K(Xnn, i=self.kk)
        # X_aug = tf.concat([tf.concat([Xnn, tf.ones(shape=[tf.shape(Xnn)[0], 1], dtype=settings.float_type) * i], axis=1) for i in range(self.Mo_dim)], axis=0)
        # K = self.kern.kernels[1].K(Xnn)
        # K = self.kern.K(X_aug)
        # return Xnn, K
        XB = tf.concat([tf.concat([tf.zeros(shape=[1, tf.shape(Xnn)[1]], dtype=settings.float_type),
                                   tf.ones(shape=[1, 1], dtype=settings.float_type) * i], axis=1) for i in
                        range(self.Mo_dim)], axis=0)
        B = self.kern.K(XB, i=self.kb)
        # B = tf.matmul(self.kern.kernels[0].W.parameter_tensor, self.kern.kernels[0].W.parameter_tensor, transpose_b=True) + tf.matrix_diag(
        #     self.kern.kernels[0].kappa.parameter_tensor)

        N = tf.shape(self.X)[0]
        y_vec = tf.reshape(tf.transpose(self.X), shape=[self.Mo_dim * N, 1])
        m = self.mean_function(y_vec)
        X_vec = y_vec - m

        l_k, q_k = tf.linalg.eigh(K)
        l_b, q_b = tf.linalg.eigh(B)

        QbQkX_vec = self.mat_vec_mul(tf.transpose(q_b), tf.transpose(q_k), X_vec)
        kron_diag = tf.concat([l_k[:, None] * l_b[i] for i in range(self.Mo_dim)], axis=0)

        Inv_vec = QbQkX_vec / (kron_diag + self.likelihood.variance)

        with tf.control_dependencies([tf.check_numerics(Inv_vec, 'inv not nan')]):     # needs to go through the checking to continue
            Inv_vec = tf.identity(Inv_vec)

        alpha = self.mat_vec_mul(q_b, q_k, Inv_vec)
        # return QbQkX_vec, Inv_vec, alpha              # tested

        logpdf = -0.5 * tf.matmul(X_vec, alpha, transpose_a=True)
        logpdf -= 0.5 * tf.cast(tf.shape(X_vec)[0], dtype=settings.float_type) * np.log(2 * np.pi)
        # logpdf -= 0.5 * tf.reduce_sum(tf.log(kron_diag_nn + self.likelihood.variance))
        logpdf -= 0.5 * tf.reduce_sum(tf.log(kron_diag + self.likelihood.variance))

        return K, B, y_vec, l_k, q_k, l_b, q_b, QbQkX_vec, kron_diag, Inv_vec, alpha

    def mat_vec_mul(self, B, K, X_vec):
        '''
        Efficient matrix-vector multiplication where
        the matrix is Kronecker product $B \cross K$,
        and the vector is $X_vec$
        :param B: First term Kronecker product
        :param K: Second term Kronecker product
        :param X_vec: vector of shape (Gb*Gk, 1)
        :return: matmul(Kron(B, K), X_vec)
        '''

        Gb = tf.shape(B)[0]
        Gk = tf.shape(K)[1]

        X_Gk = tf.reshape(X_vec, shape=[Gb, Gk])
        Z = tf.matmul(X_Gk, K, transpose_b=True)
        Z_vec = tf.reshape(tf.transpose(Z), shape=[Gb * Gk, 1])

        Z_Gb = tf.reshape(Z_vec, shape=[Gk, Gb])
        M = tf.matmul(Z_Gb, B, transpose_b=True)
        # x_out = tf.reshape(tf.transpose(M), shape=[Gb * Gk, 1])
        x_out = tf.reshape(tf.transpose(M), shape=[-1, 1])
        return x_out


    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Xnn = self.nn.forward(self.X)
        y = self.Y - self.mean_function(Xnn)
        Kmn = self.kern.K(Xnn, Xnew, i=self.km)
        Kmm_sigma = self.kern.K(Xnn, i=self.km) + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * (
                    self.likelihood.variance + settings.numerics.jitter_level)

        Knn = self.kern.K(Xnew, i=self.km) if full_cov else self.kern.Kdiag(Xnew, i=self.km)

        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_function(Xnew), f_var


    @params_as_tensors
    def _build_predict_x(self, Xnew, full_cov=False):
        Xnn = self.nn.forward(self.X)

        K = self.kern.K(Xnn, i=self.kk)
        XB = tf.concat([tf.concat([tf.zeros(shape=[1, tf.shape(Xnn)[1]], dtype=settings.float_type),
                                   tf.ones(shape=[1, 1], dtype=settings.float_type) * i], axis=1) for i in
                        range(self.Mo_dim)], axis=0)
        B = self.kern.K(XB, i=self.kb)

        N = tf.shape(self.X)[0]
        y_vec = tf.reshape(tf.transpose(self.X), shape=[self.Mo_dim * N, 1])
        m = self.mean_function(y_vec)
        X_vec = y_vec - m

        l_k, q_k = tf.linalg.eigh(K)
        l_b, q_b = tf.linalg.eigh(B)

        QbQkX_vec = self.mat_vec_mul(tf.transpose(q_b), tf.transpose(q_k), X_vec)
        kron_diag = tf.concat([l_k[:, None] * l_b[i] for i in range(self.Mo_dim)], axis=0)

        Inv_vec = QbQkX_vec / (kron_diag + self.likelihood.variance)

        alpha = self.mat_vec_mul(q_b, q_k, Inv_vec)

        Kmn = self.kern.K(Xnew, Xnn, i=self.kk)
        f_mean = self.mat_vec_mul(B, Kmn, alpha)
        # return tf.transpose(tf.reshape(f_mean, shape=[self.Mo_dim, tf.shape(Xnew)[0]]))     # , self.mean_function(tf.tile(Xnew, multiples=[self.Mo_dim, 1])), y_vec, self.X, tf.transpose(tf.reshape(y_vec, shape=[self.Mo_dim, tf.shape(self.X)[0]]))


        # posterior covariance matrix !! For a single test point !!
        Kmm = self.kern.K(Xnew, i=self.kk)
        BLin = tf.linalg.LinearOperatorFullMatrix(B)
        KmmLin = tf.linalg.LinearOperatorFullMatrix(Kmm)
        kss = tf.linalg.LinearOperatorKronecker([BLin, KmmLin]).to_dense()

        KnmLin = tf.linalg.LinearOperatorFullMatrix(tf.transpose(Kmn))
        Ks = tf.linalg.LinearOperatorKronecker([BLin, KnmLin]).to_dense()

        # QbQkKs = tf.concat(
        #     [self.mat_vec_mul(tf.transpose(q_b), tf.transpose(q_k), Ks[:, i][:, None]) for i in range(self.Mo_dim)],
        #     axis=1)
        # Resc_mat = tf.concat(
        #     [QbQkKs[:, i][:, None] / (kron_diag + self.likelihood.variance) for i in range(self.Mo_dim)], axis=1)
        # Inv_mat = tf.concat([self.mat_vec_mul(q_b, q_k, Resc_mat[:, i][:, None]) for i in range(self.Mo_dim)], axis=1)
        # mat_cov = tf.concat([self.mat_vec_mul(B, Kmn, Inv_mat[:, i][:, None]) for i in range(self.Mo_dim)], axis=1)
        mat_cov = tf.concat(
            [self.mat_vec_mul(B, Kmn, self.mat_vec_mul(q_b, q_k, self.mat_vec_mul(tf.transpose(q_b), tf.transpose(q_k),
                                                                                  Ks[:, i][:, None]) / (
                                                                   kron_diag + self.likelihood.variance))) for i in
             range(self.Mo_dim)],
            axis=1)

        # fmean_test = self.mat_vec_mul_test(B, Kmn, alpha)
        # test = self.mat_vec_mul(B, Kmn, Inv_mat[:, 0][:, None])
        # test = self.mat_vec_mul(B, Kmn, Inv_mat[:, 0][:, None])
        post_cov = kss - mat_cov
        # return f_mean, alpha, Inv_mat[:, 0][:, None], Inv_mat, B, Kmn, fmean_test, test
        return f_mean + self.mean_function(tf.tile(Xnew, multiples=[self.Mo_dim, 1])), post_cov     # , kss - mat_cov_test

    # def mat_vec_mul_test(self, B, K, X_vec):
    #     '''
    #     Efficient matrix-vector multiplication where
    #     the matrix is Kronecker product $B \cross K$,
    #     and the vector is $X_vec$
    #     :param B: First term Kronecker product
    #     :param K: Second term Kronecker product
    #     :param X_vec: vector of shape (Gb*Gk, 1)
    #     :return: matmul(Kron(B, K), X_vec)
    #     '''
    #
    #     Gb = tf.shape(B)[0]
    #     Gk = tf.shape(K)[0]
    #
    #     X_Gk = tf.reshape(X_vec, shape=[Gb, Gk])
    #     Z = tf.matmul(X_Gk, K, transpose_b=True)
    #     Z_vec = tf.reshape(tf.transpose(Z), shape=[Gb * Gk, 1])
    #
    #     Z_Gb = tf.reshape(Z_vec, shape=[Gk, Gb])
    #     M = tf.matmul(Z_Gb, B, transpose_b=True)
    #     x_out = tf.reshape(tf.transpose(M), shape=[Gb * Gk, 1])
    #     return x_out
    #     # return Z


    @params_as_tensors
    def _build_sample_x(self, Xnew):

        # Sample from GP prior assuming no noise
        Xnn = self.nn.forward(self.X)
        X_all = tf.concat([Xnew, Xnn], axis=0)
        K_all = self.kern.K(X_all, i=self.kk)

        XB = tf.concat([tf.concat([tf.zeros(shape=[1, tf.shape(Xnn)[1]], dtype=settings.float_type),
                                   tf.ones(shape=[1, 1], dtype=settings.float_type) * i], axis=1) for i in
                        range(self.Mo_dim)], axis=0)
        B = self.kern.K(XB, i=self.kb)

        # LK_all = tf.linalg.cholesky(
        #     K_all + tf.eye(tf.shape(K_all)[0], dtype=settings.float_type) * settings.numerics.jitter_level)
        # LB = tf.linalg.cholesky(B + tf.eye(tf.shape(B)[0], dtype=settings.float_type) * settings.numerics.jitter_level)

        z = tf.random.normal(shape=[tf.shape(K_all)[0] * self.Mo_dim, 1], mean=0.0, stddev=1.0,
                             dtype=settings.float_type, seed=None, name=None)

        m_prior = tf.concat([self.mean_function(tf.tile(Xnew, multiples=[self.Mo_dim, 1])),
                             self.mean_function(tf.tile(Xnn, multiples=[self.Mo_dim, 1]))], axis=0)
        # prior_sample = self.mat_vec_mul(LB, LK_all, z) + m_prior        # check shape
        #
        # LK_all_inv = tf.matrix_triangular_solve(LK_all, tf.eye(tf.shape(LK_all)[0], dtype=settings.float_type),
        #                                         lower=True, adjoint=False)
        # LB_inv = tf.matrix_triangular_solve(LB, tf.eye(tf.shape(LB)[0], dtype=settings.float_type), lower=True,
        #                                     adjoint=False)
        # K_all_inv = tf.matmul(LK_all_inv, LK_all_inv, transpose_a=True)
        # B_inv = tf.matmul(LB_inv, LB_inv, transpose_a=True)

        l_k, q_k = tf.linalg.eigh(K_all)
        # np.matmul(np.multiply(q_k, l_k), np.transpose(q_k)) = K_all
        l_b, q_b = tf.linalg.eigh(B)

        kron_diag = tf.concat([l_k[:, None] * l_b[i] for i in range(self.Mo_dim)], axis=0)
        kron_diag0 = tf.where(tf.less_equal(kron_diag, tf.zeros_like(kron_diag)), tf.zeros_like(kron_diag), kron_diag)

        prior_sample_test = self.mat_vec_mul(q_b, q_k, tf.multiply(z, tf.sqrt(
            kron_diag0 + settings.numerics.jitter_level))) + m_prior
        # return prior_sample, prior_sample_test, l_k, q_k, l_b, q_b, LK_all, LB, B, K_all, z
        # return prior_sample, prior_sample_test, K_all, K_all_inv, B, B_inv, l_k, q_k, l_b, q_b

        # obtain the posterior sample from prior sample
        K = self.kern.K(Xnn, i=self.kk)
        # LK = tf.linalg.cholesky(K + tf.eye(tf.shape(K)[0], dtype=settings.float_type) * settings.numerics.jitter_level)
        #
        # LK_inv = tf.matrix_triangular_solve(LK, tf.eye(tf.shape(LK)[0], dtype=settings.float_type), lower=True,
        #                                     adjoint=False)
        # LB_inv = tf.matrix_triangular_solve(LB, tf.eye(tf.shape(LB)[0], dtype=settings.float_type), lower=True,
        #                                     adjoint=False)  # check that: (LK_inv)T(LK_inv) = inv(K), and same for B
        #
        # K_inv = tf.matmul(LK_inv, LK_inv, transpose_a=True)
        # B_inv = tf.matmul(LB_inv, LB_inv, transpose_a=True)
        # return B, K, LB, LK, LB_inv, LK_inv, B_inv, K_inv

        N = tf.shape(self.X)[0]
        y_vec = tf.reshape(tf.transpose(self.X), shape=[self.Mo_dim * N, 1])  # observations

        M = tf.shape(Xnew)[0]
        indicesY = tf.concat(
            [tf.range(start=(i + 1) * M + i * N, limit=(i + 1) * M + (i + 1) * N, dtype=settings.int_type) for i in
             range(self.Mo_dim)], axis=0)
        indicesX = tf.concat([tf.range(start=i * M + i * N, limit=(i + 1) * M + i * N, dtype=settings.int_type) for i in
                              range(self.Mo_dim)], axis=0)
        # return indicesY, indicesX
        # Y_vec = tf.gather(prior_sample, indicesY, axis=0)  # samples
        # X_vec = tf.gather(prior_sample, indicesX, axis=0)
        # return Y_vec, X_vec
        Y_vec_test = tf.gather(prior_sample_test, indicesY, axis=0)
        X_vec_test = tf.gather(prior_sample_test, indicesX, axis=0)

        # inv_vec = self.mat_vec_mul(B_inv, K_inv,
        #                            y_vec - Y_vec)  # http://www.stats.ox.ac.uk/~doucet/doucet_simulationconditionalgaussian.pdf

        l_K, q_K = tf.linalg.eigh(K)
        kron_diag_BK = tf.concat([l_K[:, None] * l_b[i] for i in range(self.Mo_dim)], axis=0)
        kron_diag_BK0 = tf.where(tf.less_equal(kron_diag_BK, tf.zeros_like(kron_diag_BK)), tf.zeros_like(kron_diag_BK),
                                 kron_diag_BK)
        inv_vec_test = self.mat_vec_mul(q_b, q_K,
                                        self.mat_vec_mul(tf.transpose(q_b), tf.transpose(q_K), y_vec - Y_vec_test) / (
                                                    kron_diag_BK0 + settings.numerics.jitter_level))

        # return y_vec, Y_vec, y_vec - Y_vec
        Kmn = self.kern.K(Xnew, Xnn, i=self.kk)

        # update = self.mat_vec_mul(B, Kmn, inv_vec)
        update_test = self.mat_vec_mul(B, Kmn, inv_vec_test)
        # posterior_sample = X_vec + update
        posterior_sample_test = X_vec_test + update_test
        # return tf.transpose(tf.reshape(tf.transpose(posterior_sample), [tf.shape(self.X)[1], tf.shape(Xnew)[0]])), tf.transpose(tf.reshape(tf.transpose(posterior_sample_test), [tf.shape(self.X)[1], tf.shape(Xnew)[0]]))
        return tf.transpose(tf.reshape(tf.transpose(posterior_sample_test), [tf.shape(self.X)[1], tf.shape(Xnew)[0]]))
        # # return tf.reshape(posterior_sample, [tf.shape(Xnew)[0], tf.shape(self.X)[1]])


class alpha(Parameterized):
    def __init__(self, name=None):
        # super().__init__(self)
        Parameterized.__init__(self)
        np.random.seed(123)
        setattr(self, 'value', Parameter(value=np.ones(shape=[1, 1]), transform=transforms.positive,
                                         prior=None, trainable=True, dtype=settings.float_type, fix_shape=True,
                                         name=name))

    @params_as_tensors
    def forward(self):
        W = getattr(self, 'value')
        return W


class NN_BLRMoGP(GPModel):
    """
    Define a prior over the linear mapping A (AA^T = Coregionalization matrix of Multi-output GP), A (D x p matrix)
    Derive the posterior of A given GP samples "u(Z)", and observations "g(Z)"
    Compute the marginal likelihood marginalizing over the posterior over "A"
    Comute posterior distribution for each test point marginalizing over the posterior over "A"

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, nn, Mo_dim, alpha, sample_train, sample_test, p=None, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.nn = nn
        self.Mo_dim = Mo_dim
        self.p = X.shape[0] if p is None else p
        self.N = X.shape[0]
        self.standard_train = DataHolder(sample_train)
        self.standard_test = DataHolder(sample_test)
        # self.standard_train = tf.random.normal(shape=[self.N, self.p], mean=0.0, stddev=1.0, dtype=settings.float_type,
        #                                        seed=None, name=None)
        self.priorA = {'M': tf.zeros(shape=[self.Mo_dim, self.p], dtype=settings.float_type),
                       'V': tf.eye(self.Mo_dim, dtype=settings.float_type), 'K': None}      # A ~ N(M, V, K), M(Dxp), V(DxD), K(pxp)
        self.alpha = alpha
        # Assuming kern in input is "Multiple_k" kernel with:
        # kern.K(, i=0) = gpflow.kernels.Matern52/RBF/etc.  Standard kernel MOGP
        # kern.K(, i=1) = gpflow.kernels.Matern52/RBF/etc.  Standard kernel Manifold GP
        self.kk = int(0)    # index MOGP kernel
        self.km = int(1)    # index Manifold GP kernel
        self.uZ = sample_train
        self.jitter = Parameter(1e-05, transform=transforms.positive, prior=None, trainable=False, dtype=settings.float_type, fix_shape=True)
        self.noise_variance = Parameter(1e-05, transform=transforms.positive, prior=None, trainable=False, dtype=settings.float_type, fix_shape=True)


    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """

        Xnn = self.nn.forward(self.X)

        # sample from the GP prior
        KZZ = self.kern.K(Xnn, i=self.kk)
        # KZZ = tf.Print(KZZ, [KZZ])
        # LZZ = tf.linalg.cholesky(
        #     KZZ + tf.eye(tf.shape(KZZ)[0], dtype=settings.float_type) * self.jitter)
        # self.uZ = tf.matmul(self.standard_train, LZZ, transpose_a=True, transpose_b=True) + tf.tile(
        #     tf.transpose(self.mean_function(Xnn)), multiples=[self.p, 1])
        e_KZZ, v_KZZ = tf.linalg.eigh(KZZ)
        e_KZZ05 = tf.sqrt(e_KZZ + self.jitter)
        self.uZ = tf.matmul(self.standard_train, tf.matmul(v_KZZ, tf.matrix_diag(e_KZZ05)), transpose_a=True, transpose_b=True) + tf.tile(
            tf.transpose(self.mean_function(Xnn)), multiples=[self.p, 1])

        # set invariant prior for "A"
        self.priorA['K'] = self.alpha.forward() * tf.matmul(self.uZ, self.uZ, transpose_b=True)   # invariant prior with alpha positive parameter

        # invert prior covariance matrix "K" over the columns of "A"
        LK = tf.linalg.cholesky(
            self.priorA['K'] + tf.eye(tf.shape(self.priorA['K'])[0], dtype=settings.float_type) * self.jitter)
        LK_inv = tf.matrix_triangular_solve(LK, tf.eye(tf.shape(LK)[0], dtype=settings.float_type), lower=True,
                                            adjoint=False)
        Kprior_inv = tf.matmul(LK_inv, LK_inv, transpose_a=True)


        # marginal likelihood components
        SMarg = tf.matmul(self.uZ, tf.matmul(Kprior_inv, self.uZ), transpose_a=True) + tf.eye(tf.shape(self.X)[0],
                                                                                    dtype=settings.float_type) * self.noise_variance
        MMarg = tf.matmul(self.priorA['M'], self.uZ)
        # True Distribution: N(vec(MMarg), SMarg \kron I_{self.Mo_dim})
        # SMarg_wo_noise = tf.matmul(self.uZ, tf.matmul(Kprior_inv, self.uZ), transpose_a=True)
        # e_SMarg, v_SMarg = tf.linalg.eigh(SMarg_wo_noise)


        gZ = tf.transpose(self.X)   # D x N
        vec_gZ = self.vectorize(gZ)
        vec_MMarg = self.vectorize(MMarg)
        X_vec = vec_gZ - vec_MMarg

        LMarg = tf.linalg.cholesky(SMarg)
        LMarg_inv = tf.matrix_triangular_solve(LMarg, tf.eye(tf.shape(LMarg)[0], dtype=settings.float_type), lower=True,
                                               adjoint=False)
        SMarg_inv = tf.matmul(LMarg_inv, LMarg_inv, transpose_a=True)
        # SMarg_inv = tf.matmul(v_SMarg, tf.matmul(tf.matrix_diag(1. / (e_SMarg + self.noise_variance)), v_SMarg), transpose_a=True)

        LV = tf.linalg.cholesky(self.priorA['V'])
        LV_inv = tf.matrix_triangular_solve(LV, tf.eye(tf.shape(LV)[0], dtype=settings.float_type), lower=True,
                                            adjoint=False)
        V_inv = tf.matmul(LV_inv, LV_inv, transpose_a=True)

        inv_vec = self.mat_vec_mul(SMarg_inv, V_inv, X_vec)

        logpdf = - 0.5 * tf.matmul(X_vec, inv_vec, transpose_a=True)
        logpdf -= 0.5 * tf.cast(tf.shape(X_vec)[0], dtype=settings.float_type) * np.log(2.0 * np.pi)
        logpdf -= 0.5 * (tf.cast(tf.shape(self.priorA['V'])[0], dtype=settings.float_type) * tf.linalg.logdet(
            SMarg) + tf.cast(tf.shape(SMarg)[0], dtype=settings.float_type) * tf.linalg.logdet(self.priorA['V']))

        # logpdf -= 0.5 * (tf.cast(tf.shape(self.priorA['V'])[0], dtype=settings.float_type) * tf.reduce_sum(
        #     tf.log(e_SMarg + self.noise_variance)) + tf.cast(tf.shape(SMarg)[0],
        #                                                      dtype=settings.float_type) * tf.linalg.logdet(
        #     self.priorA['V']))


        # Marginal likelihood contribution of the Manifold GP
        # Km = self.kern.K(Xnn, i=self.km) + tf.eye(tf.shape(Xnn)[0],
        #                                           dtype=settings.float_type) * (self.likelihood.variance + self.jitter)    # + settings.numerics.jitter_level)
        Km = self.kern.K(Xnn, i=self.km)
        # Km = tf.Print(Km, [Km])
        # Lm = tf.cholesky(Km)
        mm = self.mean_function(Xnn)
        d = self.Y - mm
        # logpdfm = multivariate_normal(self.Y, mm, Lm)
        e_Km, v_Km = tf.linalg.eigh(Km)
        logpdfm = - 0.5 * tf.cast(tf.shape(Xnn)[0], dtype=settings.float_type) * np.log(2.0 * np.pi)
        logpdfm -= 0.5 * tf.reduce_sum(tf.log(e_Km + self.likelihood.variance + self.jitter))
        logpdfm -= 0.5 * tf.matmul(d, tf.matmul(tf.matmul(v_Km, tf.matmul(
            tf.matrix_diag(1. / (e_Km + self.likelihood.variance + self.jitter)), v_Km, transpose_b=True)), d),
                                   transpose_a=True)
        return logpdf[0][0] + logpdfm[0][0]
        # return logpdf
        # return logpdf[0][0] + tf.reduce_sum(logpdfm)


    @params_as_tensors
    def _test_likelihood(self, Xnew):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        Xnn = self.nn.forward(self.X)

        # sample from the GP prior
        KZZ = self.kern.K(Xnn, i=self.kk)
        # LZZ = tf.linalg.cholesky(
        #     KZZ + tf.eye(tf.shape(KZZ)[0], dtype=settings.float_type) * self.jitter)
        # self.uZ = tf.matmul(self.standard_train, LZZ, transpose_a=True, transpose_b=True) + tf.tile(
        #     tf.transpose(self.mean_function(Xnn)), multiples=[self.p, 1])
        e_KZZ, v_KZZ = tf.linalg.eigh(KZZ)
        e_KZZ05 = tf.sqrt(e_KZZ + self.jitter)
        self.uZ = tf.matmul(self.standard_train, tf.matmul(v_KZZ, tf.matrix_diag(e_KZZ05)), transpose_a=True, transpose_b=True) + tf.tile(
            tf.transpose(self.mean_function(Xnn)), multiples=[self.p, 1])
        # check repeated after multiple times computed      ok

        # obtain uZ from the prior with Kronecker product in the covariance matrix: Cov(vec(uZ)^T) = I_{self.p} \kron KZZ
        Lp = tf.linalg.cholesky(tf.eye(self.p, dtype=settings.float_type))
        # uZ_test = self.mat_vec_mul(Lp, LZZ, self.vectorize(self.standard_train))
        uZ_test = self.mat_vec_mul(Lp, tf.matmul(v_KZZ, tf.matrix_diag(e_KZZ05)), self.vectorize(self.standard_train))
        uZ_test_reshape = tf.reshape(uZ_test, shape=[self.p, tf.shape(Xnn)[0]]) + tf.tile(
            tf.transpose(self.mean_function(Xnn)), multiples=[self.p, 1])
        # check "uZ_test_reshape == uZ"

        Lk = tf.linalg.cholesky(self.kern.K(Xnew, i=self.kk) + tf.eye(tf.shape(Xnew)[0],
                                                                      dtype=settings.float_type) * self.jitter)
        std_vec = tf.random.normal(shape=[tf.shape(Xnew)[0], self.p], mean=0.0, stddev=1.0, dtype=settings.float_type,
                                   seed=None, name=None)
        uZnew = tf.matmul(std_vec, Lk, transpose_a=True, transpose_b=True) + tf.tile(
            tf.transpose(self.mean_function(Xnew)), multiples=[self.p, 1])
        # plot the rows and the columns, should observe continuous functions over columns i=1,...,N, should observe uncorrelated noise over the rows j=1,...,p

        # set invariant prior for "A"
        self.priorA['K'] = self.alpha.forward() * tf.matmul(self.uZ, self.uZ, transpose_b=True)   # invariant prior with alpha positive parameter

        # invert prior covariance matrix "K" over the columns of "A"
        LK = tf.linalg.cholesky(
            self.priorA['K'] + tf.eye(tf.shape(self.priorA['K'])[0], dtype=settings.float_type) * self.jitter)
        LK_inv = tf.matrix_triangular_solve(LK, tf.eye(tf.shape(LK)[0], dtype=settings.float_type), lower=True,
                                            adjoint=False)
        Kprior_inv = tf.matmul(LK_inv, LK_inv, transpose_a=True)
        # check "Kprior_inv == self.priorA['K']^{-1}"

        # marginal likelihood components
        SMarg = tf.matmul(self.uZ, tf.matmul(Kprior_inv, self.uZ), transpose_a=True) + tf.eye(tf.shape(self.X)[0],
                                                                                    dtype=settings.float_type) * self.noise_variance
        MMarg = tf.matmul(self.priorA['M'], self.uZ)
        # True Distribution: N(vec(MMarg), SMarg \kron I_{self.Mo_dim})

        gZ = tf.transpose(self.X)   # D x N
        vec_gZ = self.vectorize(gZ)
        vec_MMarg = self.vectorize(MMarg)
        X_vec = vec_gZ - vec_MMarg
        check_same_as_X_vec = self.vectorize(gZ - MMarg)
        # check "check_same_as_X_vec == X_vec"

        LMarg = tf.linalg.cholesky(SMarg)
        LMarg_inv = tf.matrix_triangular_solve(LMarg, tf.eye(tf.shape(LMarg)[0], dtype=settings.float_type), lower=True,
                                               adjoint=False)
        SMarg_inv = tf.matmul(LMarg_inv, LMarg_inv, transpose_a=True)
        # check "SMarg_inv == SMarg^{-1}"

        LV = tf.linalg.cholesky(self.priorA['V'])
        LV_inv = tf.matrix_triangular_solve(LV, tf.eye(tf.shape(LV)[0], dtype=settings.float_type), lower=True,
                                            adjoint=False)
        V_inv = tf.matmul(LV_inv, LV_inv, transpose_a=True)
        # check "V_inv == V^{-1}"

        inv_vec = self.mat_vec_mul(SMarg_inv, V_inv, X_vec)

        logpdf = - 0.5 * tf.matmul(X_vec, inv_vec, transpose_a=True)
        logpdf -= 0.5 * tf.cast(tf.shape(X_vec)[0], dtype=settings.float_type) * np.log(2.0 * np.pi)
        logpdf -= 0.5 * (tf.cast(tf.shape(self.priorA['V'])[0], dtype=settings.float_type) * tf.linalg.logdet(
            SMarg) + tf.cast(tf.shape(SMarg)[0], dtype=settings.float_type) * tf.linalg.logdet(self.priorA['V']))


        # Marginal likelihood contribution of the Manifold GP
        Km = self.kern.K(Xnn, i=self.km) + tf.eye(tf.shape(Xnn)[0],
                                                  dtype=settings.float_type) * (self.likelihood.variance + self.jitter)     # + settings.numerics.jitter_level)
        # Km = tf.Print(Km, [Km])
        Lm = tf.cholesky(Km)
        mm = self.mean_function(Xnn)
        logpdfm = multivariate_normal(self.Y, mm, Lm)
        # return logpdf
        Km_test = self.kern.K(Xnn, i=self.km)
        d = self.Y - mm
        e_Km, v_Km = tf.linalg.eigh(Km_test)
        logpdfm_test = - 0.5 * tf.cast(tf.shape(Xnn)[0], dtype=settings.float_type) * np.log(2.0 * np.pi)
        logpdfm_test -= 0.5 * tf.reduce_sum(tf.log(e_Km + self.likelihood.variance + self.jitter))
        logpdfm_test -= 0.5 * tf.matmul(d, tf.matmul(tf.matmul(v_Km, tf.matmul(
            tf.matrix_diag(1. / (e_Km + self.likelihood.variance + self.jitter)), v_Km, transpose_b=True)), d),
                                   transpose_a=True)
        return logpdf[0][0], tf.reduce_sum(logpdfm), self.uZ, uZ_test_reshape, uZnew, Kprior_inv, self.priorA['K'] + tf.eye(
            tf.shape(self.priorA['K'])[0],
            dtype=settings.float_type) * self.jitter, check_same_as_X_vec, X_vec, SMarg_inv, SMarg, V_inv, self.priorA['V'], tf.transpose(LK_inv), logpdfm_test, e_Km, v_Km, Km_test, d, Km, Lm



    def vectorize(self, Xin):
        D = tf.shape(Xin)[0]
        N = tf.shape(Xin)[1]
        return tf.reshape(tf.transpose(Xin), shape=[D * N, 1])


    def mat_vec_mul(self, B, K, X_vec):
        '''
        Efficient matrix-vector multiplication where
        the matrix is Kronecker product $B \cross K$,
        and the vector is $X_vec$
        :param B: First term Kronecker product
        :param K: Second term Kronecker product
        :param X_vec: vector of shape (Gb*Gk, 1)
        :return: matmul(Kron(B, K), X_vec)
        '''

        Gb = tf.shape(B)[0]
        Gk = tf.shape(K)[1]

        X_Gk = tf.reshape(X_vec, shape=[Gb, Gk])
        Z = tf.matmul(X_Gk, K, transpose_b=True)
        Z_vec = tf.reshape(tf.transpose(Z), shape=[Gb * Gk, 1])

        Z_Gb = tf.reshape(Z_vec, shape=[Gk, Gb])
        M = tf.matmul(Z_Gb, B, transpose_b=True)
        # x_out = tf.reshape(tf.transpose(M), shape=[Gb * Gk, 1])
        x_out = tf.reshape(tf.transpose(M), shape=[-1, 1])
        return x_out


    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        """
        Xnn = self.nn.forward(self.X)
        y = self.Y - self.mean_function(Xnn)
        Kmn = self.kern.K(Xnn, Xnew, i=self.km)
        Kmm_sigma = self.kern.K(Xnn, i=self.km) + tf.eye(tf.shape(Xnn)[0],
                                                  dtype=settings.float_type) * (self.likelihood.variance + self.jitter)

        Knn = self.kern.K(Xnew, i=self.km) if full_cov else self.kern.Kdiag(Xnew, i=self.km)

        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_function(Xnew), f_var


    @params_as_tensors
    def _build_predict_x(self, Xnew, full_cov=False):
        '''

        :param Xnew:
        :param full_cov:
        :return: post_mean, dict with posterior covariance factors: 'Kpost', 'V'. The full covariance of Vec(post_mean) is : Kpost \kron V
        !!! WARNING !!! Assuming fixed standard normal for test_set, "_build_predict_x" can only predict for a single input
        '''
        Xnn = self.nn.forward(self.X)

        y = tf.transpose(self.uZ - tf.tile(tf.transpose(self.mean_function(Xnn)), multiples=[self.p, 1]))
        Kmn = self.kern.K(Xnn, Xnew, i=self.kk)
        Kmm_sigma = self.kern.K(Xnn, i=self.kk) + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * 1e-09
        Knn = self.kern.K(Xnew, i=self.kk)
        uZsT_mean, uZsT_cov = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=True, white=False)  # N x P, N x P or P x N x N
        # return uZsT_cov
        Lpost = tf.linalg.cholesky(uZsT_cov[0, :, :] + tf.eye(tf.shape(uZsT_cov)[-1], dtype=settings.float_type) * 1e-09)
        uZs = tf.transpose(uZsT_mean + tf.matmul(Lpost, self.standard_test))
        # e_cov, v_cov = tf.linalg.eigh(uZsT_cov[0, :, :])
        # e_cov05 = tf.sqrt(e_cov + 1e-09)
        # uZs = tf.transpose(uZsT_mean + tf.matmul(tf.matmul(v_cov, tf.matrix_diag(e_cov05)), self.standard_test))
        # uZs = tf.transpose(uZsT_mean)
        # return f_mean + self.mean_function(Xnew), f_var


        # X_all = tf.concat([Xnew, Xnn], axis=0)
        # K_all = self.kern.K(X_all, i=self.kk)
        # L_all = tf.linalg.cholesky(
        #     K_all + tf.eye(tf.shape(K_all)[0], dtype=settings.float_type) * self.jitter)
        # # standard_all = tf.concat([tf.random.normal(shape=[tf.shape(Xnew)[0], self.p], mean=0.0, stddev=1.0,
        # #                                            dtype=settings.float_type, seed=None, name=None),
        # #                           self.standard_train], axis=0)
        # standard_all = tf.concat([self.standard_test, self.standard_train], axis=0)
        #
        # uZsuZ = tf.matmul(standard_all, L_all, transpose_a=True, transpose_b=True) + tf.tile(
        #     tf.transpose(self.mean_function(X_all)), multiples=[self.p, 1])     # p x (M + N)
        #
        # indices_uZs = tf.range(start=0, limit=tf.shape(Xnew)[0], dtype=settings.int_type)
        # uZs = tf.gather(uZsuZ, indices=indices_uZs, axis=1)
        #
        # indices_uZ = tf.range(start=tf.shape(Xnew)[0], limit=tf.shape(Xnew)[0] + tf.shape(Xnn)[0],
        #                        dtype=settings.int_type)
        # uZ = tf.gather(uZsuZ, indices=indices_uZ, axis=1)


        # posterior over "A", noise variance is assumed to be settings.numerics.jitter_level = noise.variance
        gZ = tf.transpose(self.X)       # D x N
        Sxx = (1. / self.noise_variance) * tf.matmul(self.uZ, self.uZ, transpose_b=True) + self.priorA['K']    # posterior covariance of "A" with noise
        LSxx = tf.linalg.cholesky(
            Sxx + tf.eye(tf.shape(Sxx)[0], dtype=settings.float_type) * self.jitter)
        LSxx_inv = tf.matrix_triangular_solve(LSxx, tf.eye(tf.shape(LSxx)[0], dtype=settings.float_type), lower=True,
                                              adjoint=False)
        Sxx_inv = tf.matmul(LSxx_inv, LSxx_inv, transpose_a=True)   # = SN
        MN = tf.matmul(
            (1. / self.noise_variance) * tf.matmul(gZ, self.uZ, transpose_b=True) + tf.matmul(self.priorA['M'],
                                                                                                    self.priorA['K']),
            Sxx_inv)
        # True Formula: N(vec(MN), SN = Sxx_inv \kron I_{self.Mo_dim})

        # posterior predictions
        post_mean = tf.matmul(MN, uZs)      # D x M

        post_S = tf.matmul(uZs, tf.matmul(Sxx_inv, uZs), transpose_a=True)      # no noise because it is prediction

        # the posterior covariance matrix for the vectorized posterior mean is the kronecker product: "post_s \kron self.priorA['V']"
        return tf.transpose(post_mean), post_S, self.priorA['V']        # posterior distribution: N(vec(post_mean), self.priorA['V'] \kron post_s)
        # return post_mean, {'Kpost': post_S, 'V': self.priorA['V'],
        #                    'Instructions': 'full covariance of vec(post_mean) is : Kpost \kron V'}  # ok


    @params_as_tensors
    def _test_predict_x(self, Xnew, full_cov=False):
        '''

        :param Xnew:
        :param full_cov:
        :return: post_mean, dict with posterior covariance factors: 'Kpost', 'V'. The full covariance of Vec(post_mean) is : Kpost \kron V
        !!! WARNING !!! Assuming fixed standard normal for test_set, "_build_predict_x" can only predict for a single input
        '''
        Xnn = self.nn.forward(self.X)

        y = tf.transpose(self.uZ - tf.tile(tf.transpose(self.mean_function(Xnn)), multiples=[self.p, 1]))
        Kmn = self.kern.K(Xnn, Xnew, i=self.kk)
        Kmm_sigma = self.kern.K(Xnn, i=self.kk) + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * 1e-09
        Knn = self.kern.K(Xnew, i=self.kk)
        uZsT_mean, uZsT_cov = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=True, white=False)  # N x P, N x P or P x N x N
        # return uZsT_cov
        Lpost = tf.linalg.cholesky(uZsT_cov[0, :, :] + tf.eye(tf.shape(uZsT_cov)[-1], dtype=settings.float_type) * 1e-09)
        uZs = tf.transpose(uZsT_mean + tf.matmul(Lpost, self.standard_test))
        # uZs = tf.transpose(uZsT_mean)
        # return f_mean + self.mean_function(Xnew), f_var


        # X_all = tf.concat([Xnew, Xnn], axis=0)
        # K_all = self.kern.K(X_all, i=self.kk)
        # L_all = tf.linalg.cholesky(
        #     K_all + tf.eye(tf.shape(K_all)[0], dtype=settings.float_type) * self.jitter)
        # # standard_all = tf.concat([tf.random.normal(shape=[tf.shape(Xnew)[0], self.p], mean=0.0, stddev=1.0,
        # #                                            dtype=settings.float_type, seed=None, name=None),
        # #                           self.standard_train], axis=0)
        # standard_all = tf.concat([self.standard_test, self.standard_train], axis=0)
        #
        # uZsuZ = tf.matmul(standard_all, L_all, transpose_a=True, transpose_b=True) + tf.tile(
        #     tf.transpose(self.mean_function(X_all)), multiples=[self.p, 1])     # p x (M + N)
        #
        # indices_uZs = tf.range(start=0, limit=tf.shape(Xnew)[0], dtype=settings.int_type)
        # uZs = tf.gather(uZsuZ, indices=indices_uZs, axis=1)
        #
        # indices_uZ = tf.range(start=tf.shape(Xnew)[0], limit=tf.shape(Xnew)[0] + tf.shape(Xnn)[0],
        #                        dtype=settings.int_type)
        # uZ = tf.gather(uZsuZ, indices=indices_uZ, axis=1)


        # posterior over "A", noise variance is assumed to be settings.numerics.jitter_level = noise.variance
        gZ = tf.transpose(self.X)       # D x N
        Sxx = (1. / self.noise_variance) * tf.matmul(self.uZ, self.uZ, transpose_b=True) + self.priorA['K']    # posterior covariance of "A" with noise
        LSxx = tf.linalg.cholesky(
            Sxx + tf.eye(tf.shape(Sxx)[0], dtype=settings.float_type) * self.jitter)
        LSxx_inv = tf.matrix_triangular_solve(LSxx, tf.eye(tf.shape(LSxx)[0], dtype=settings.float_type), lower=True,
                                              adjoint=False)
        Sxx_inv = tf.matmul(LSxx_inv, LSxx_inv, transpose_a=True)   # = SN
        MN = tf.matmul(
            (1. / self.noise_variance) * tf.matmul(gZ, self.uZ, transpose_b=True) + tf.matmul(self.priorA['M'],
                                                                                                    self.priorA['K']),
            Sxx_inv)
        # True Formula: N(vec(MN), SN = Sxx_inv \kron I_{self.Mo_dim})

        # posterior predictions
        post_mean = tf.matmul(MN, uZs)      # D x M

        post_S = tf.matmul(uZs, tf.matmul(Sxx_inv, uZs), transpose_a=True)      # no noise because it is prediction

        return tf.transpose(LSxx_inv), tf.transpose(MN), tf.transpose(post_mean), post_S, self.priorA['V'], uZs, self.uZ        # posterior distribution: N(vec(post_mean), self.priorA['V'] \kron post_s)

    @params_as_tensors
    def _base_cond(self, Kmn, Kmm_sigma, Knn, y, full_cov=False, white=False):
        return base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=white)


    # @params_as_tensors
    # def _build_sample_x(self, Xnew):
    #
    #     # Sample from GP prior assuming no noise
    #     Xnn = self.nn.forward(self.X)
    #     X_all = tf.concat([Xnew, Xnn], axis=0)
    #     K_all = self.kern.K(X_all, i=self.kk)
    #     L_all = tf.linalg.cholesky(
    #         K_all + tf.eye(tf.shape(K_all)[0], dtype=settings.float_type) * 1e-04)
    #     standard_all = tf.concat([tf.random.normal(shape=[tf.shape(Xnew)[0], self.p], mean=0.0, stddev=1.0,
    #                                                dtype=settings.float_type, seed=None, name=None),
    #                               self.standard_train], axis=0)
    #
    #     uZsuZ = tf.matmul(standard_all, L_all, transpose_a=True, transpose_b=True)  # p x (M + N)
    #
    #     indices_uZs = tf.range(start=0, limit=tf.shape(Xnew)[0], dtype=settings.int_type)
    #     uZs = tf.gather(uZsuZ, indices=indices_uZs, axis=1)
    #
    #     indices_uZ = tf.range(start=tf.shape(Xnew)[0], limit=tf.shape(Xnew)[0] + tf.shape(Xnn)[0],
    #                           dtype=settings.int_type)
    #     uZ = tf.gather(uZsuZ, indices=indices_uZ, axis=1)
    #
    #
    #     prior_mean = tf.matmul(uZsuZ, self.priorA['M'], transpose_a=True, transpose_b=True)     # (M+N) x D
    #
    #
    #     LK = tf.linalg.cholesky(self.priorA['K'] + tf.eye(tf.shape(self.priorA['K'])[0],
    #                                                       dtype=settings.float_type) * 1e-04)
    #     LK_inv = tf.matrix_triangular_solve(LK, tf.eye(tf.shape(LK)[0], dtype=settings.float_type), lower=True,
    #                                           adjoint=False)
    #     K_inv = tf.matmul(LK_inv, LK_inv, transpose_a=True)
    #
    #
    #     LV = tf.linalg.cholesky(self.priorA['V'])   # V is the identity
    #     LV_inv = tf.matrix_triangular_solve(LV, tf.eye(tf.shape(LV)[0], dtype=settings.float_type), lower=True,
    #                                           adjoint=False)
    #     V_inv = tf.matmul(LV_inv, LV_inv, transpose_a=True)
    #
    #
    #     # assumed noise free
    #     cov_joint_prior = tf.matmul(uZsuZ, tf.matmul(K_inv, uZsuZ), transpose_a=True)
    #     Lcov_joint_prior = tf.linalg.cholesky(cov_joint_prior + tf.eye(tf.shape(cov_joint_prior)[0],
    #                                                                    dtype=settings.float_type) * 1e-04)
    #     LV_inv = tf.linalg.cholesky(
    #         V_inv + tf.eye(tf.shape(V_inv)[0], dtype=settings.float_type) * 1e-04)
    #
    #     standard_sample = tf.random.normal(shape=[(tf.shape(Xnew)[0] + tf.shape(Xnn)[0]) * self.Mo_dim, 1], mean=0.0,
    #                                        stddev=1.0, dtype=settings.float_type, seed=None, name=None)
    #
    #     # prior sample
    #     sample_prior = self.mat_vec_mul(LV_inv, Lcov_joint_prior, standard_sample) + self.vectorize(prior_mean)
    #     M = tf.shape(Xnew)[0]
    #     N = tf.shape(Xnn)[0]
    #     fTsfT = tf.transpose(tf.reshape(sample_prior, shape=[self.Mo_dim, M + N]))
    #     fTs = tf.gather(fTsfT, indices=indices_uZs, axis=0)     # first M are star
    #     vec_fTs = self.vectorize(fTs)
    #     fT = tf.gather(fTsfT, indices=indices_uZ, axis=0)       # last N are train
    #     vec_fT = self.vectorize(fT)
    #     vec_fT_obs = self.vectorize(self.X)
    #
    #     cov_train = tf.matmul(uZ, tf.matmul(K_inv, uZ), transpose_a=True)
    #     Lcov_train = tf.linalg.cholesky(
    #         cov_train + tf.eye(tf.shape(cov_train)[0], dtype=settings.float_type) * 1e-04)
    #     Lcov_train_inv = tf.matrix_triangular_solve(Lcov_train,
    #                                                 tf.eye(tf.shape(Lcov_train)[0], dtype=settings.float_type),
    #                                                 lower=True, adjoint=False)
    #     cov_train_inv = tf.matmul(Lcov_train_inv, Lcov_train_inv, transpose_a=True)
    #     cov_s = tf.matmul(uZs, tf.matmul(K_inv, uZ), transpose_a=True)
    #     sample_posterior = vec_fTs + self.mat_vec_mul(V_inv, cov_s, self.mat_vec_mul(self.priorA['V'], cov_train_inv,
    #                                                                                  vec_fT_obs - vec_fT))
    #     return tf.transpose(tf.reshape(sample_posterior, shape=[self.Mo_dim, M]))


class NN_MOGPR(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, nn, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.nn = nn
        self.input_dim = X.shape[1]
        # self.nn2 = nn2

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        Xnn = self.nn.forward(self.X)   # Reduce dimensionality
        # Xnn = self.nn2.forward(self.nn.forward(self.X))

        Xnn_aug = tf.concat(
            [tf.concat([Xnn, tf.ones(shape=[tf.shape(Xnn)[0], 1], dtype=settings.float_type) * i], axis=1)
             for i in range(self.input_dim + 1)], axis=0)
        Ynn_aug = tf.concat([self.X[:, i][:, None] for i in range(self.input_dim)] + [self.Y], axis=0)

        # K = self.kern.K(Xnn_aug) + tf.eye(tf.shape(Xnn_aug)[0], dtype=settings.float_type) * self.likelihood.variance
        # L = tf.cholesky(K)
        m = self.mean_function(Xnn_aug)
        # logpdf = multivariate_normal(Ynn_aug, m, L)  # (R,) log-likelihoods for each independent dimension of Y
        # logpdf = tf.Print(logpdf, [logpdf])
        # return tf.reduce_sum(logpdf)
        return self.KronLikelihood(Xnn, Ynn_aug, m, 0)
        # logpdf = self.GaBPlikelihood(Y=Ynn_aug, m=m, K=K, L=L)
        # return logpdf

    def GaBPlikelihood(self, Y, m, K, L):

        if Y.shape.ndims is None:
            logger.warn('Shape of x must be 2D at computation.')
        elif Y.shape.ndims != 2:
            raise ValueError('Shape of x must be 2D.')
        if m.shape.ndims is None:
            logger.warn('Shape of mu may be unknown or not 2D.')
        elif m.shape.ndims != 2:
            raise ValueError('Shape of mu must be 2D.')

        f = Y - m
        Kinv_f = GaBP(K, f)
        fKf = tf.matmul(f, Kinv_f, transpose_a=True)
        n = tf.cast(tf.shape(f)[0], K.dtype)
        p =  - 0.5 * n * np.log(2 * np.pi)  #  - 0.5 * fKf - 0.5 * tf.linalg.logdet(K)

        # d = Y - m
        # alpha = tf.matrix_triangular_solve(L, d, lower=True)
        # Kinv_d = tf.matmul(tf.linalg.inv(L), alpha, transpose_a=True)
        # num_dims = tf.cast(tf.shape(d)[0], L.dtype)
        # # p = - 0.5 * tf.reduce_sum(tf.square(alpha), 0)
        # p = 0.5 * num_dims * np.log(2 * np.pi)
        # # p -= tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
        # return p
        return K, f

    def KronLikelihood(self, Xnn, Y, m, logpdf):
        if Y.shape.ndims is None:
            logger.warn('Shape of x must be 2D at computation.')
        elif Y.shape.ndims != 2:
            raise ValueError('Shape of x must be 2D.')
        if m.shape.ndims is None:
            logger.warn('Shape of mu may be unknown or not 2D.')
        elif m.shape.ndims != 2:
            raise ValueError('Shape of mu must be 2D.')

        K = self.kern.kernels[0].K(Xnn) + tf.eye(tf.shape(Xnn)[0], dtype=settings.float_type) * self.likelihood.variance
        B = self.kern.kernels[1].K(tf.concat([tf.ones(shape=[self.input_dim + 1, tf.shape(Xnn)[1]], dtype=settings.float_type),
                                              tf.cast(tf.range(self.input_dim + 1)[:, None], dtype=settings.float_type)], axis=1))
        nk = tf.cast(tf.shape(K)[0], K.dtype)
        mb = tf.cast(tf.shape(B)[0], B.dtype)

        Lk = tf.cholesky(K)
        Lb = tf.cholesky(B)

        Kinv = tf.cholesky_solve(Lk, tf.eye(tf.shape(Lk)[0], dtype=settings.float_type))
        Binv = tf.cholesky_solve(Lb, tf.eye(tf.shape(Lb)[0], dtype=settings.float_type))

        d = Y - m
        num_dims = tf.cast(tf.shape(d)[0], Lk.dtype)
        p = - 0.5 * num_dims * np.log(2 * np.pi)
        # K_MATinv = tf.contrib.kfac.utils.kronecker_product(Binv, Kinv)
        operator_1 = tf.linalg.LinearOperatorFullMatrix(Binv)
        operator_2 = tf.linalg.LinearOperatorFullMatrix(Kinv)
        operator = tf.linalg.LinearOperatorKronecker([operator_1, operator_2])
        K_MATinv = operator.to_dense()
        p -= 0.5 * tf.matmul(d, tf.matmul(K_MATinv, d), transpose_a=True)
        p -= mb * tf.reduce_sum(tf.log(tf.matrix_diag_part(Lk))) + nk * tf.reduce_sum(tf.log(tf.matrix_diag_part(Lb)))

        # return K, B, Kinv, Binv, K_MATinv, p, tf.tile(logpdf[:, None], [1, tf.shape(B)[0]])
        p = tf.Print(p, [p])
        return p[:, 0]

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Xnn = self.nn.forward(self.X)
        Xnn_aug = tf.concat(
            [tf.concat([Xnn, tf.ones(shape=[tf.shape(Xnn)[0], 1], dtype=settings.float_type) * i], axis=1)
             for i in range(self.input_dim + 1)], axis=0)
        Ynn_aug = tf.concat([self.X[:, i][:, None] for i in range(self.input_dim)] + [self.Y], axis=0)

        Kx = self.kern.K(Xnn_aug, Xnew)
        K = self.kern.K(Xnn_aug) + tf.eye(tf.shape(Xnn_aug)[0], dtype=settings.float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, Ynn_aug - self.mean_function(Xnn_aug))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar



class NN_SVGP(GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews,
                Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """

    def __init__(self, X, Y, kern, likelihood, nn,
                 feat=None,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 Z=None,
                 num_data=None,
                 q_mu=None,
                 q_sqrt=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)     # num_latent = num_latent or Y.shape[1]
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)
        self.nn = nn

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = Kuu(self.feature, self.kern, jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        Xnn = self.nn.forward(self.X)
        fmean, fvar = self._build_predict(Xnn, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(Xnn)[0], settings.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(Xnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var