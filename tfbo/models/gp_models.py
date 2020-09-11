import abc

import numpy as np
import tensorflow as tf
import gpflow

from gpflow import settings
from gpflow.params import DataHolder
from gpflow.decors import autoflow
from gpflow.mean_functions import Zero


class GPModel(gpflow.models.Model):
    """
    A base class for Gaussian process models, that is, those of the form

    .. math::
       :nowrap:

       \\begin{align}
       \\theta & \sim p(\\theta) \\\\
       f       & \sim \\mathcal{GP}(m(x), k(x, x'; \\theta)) \\\\
       f_i       & = f(x_i) \\\\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \\end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.

    For handling another data (Xnew, Ynew), set the new value to self.X and self.Y

    # >>> m.X = Xnew
    # >>> m.Y = Ynew
    """

    def __init__(self, X, Y, kern, likelihood, mean_function,
                 num_latent=None, name=None):
        super(GPModel, self).__init__(name=name)
        self.num_latent = num_latent or Y.shape[1]
        self.mean_function = mean_function or Zero(output_dim=self.num_latent)
        self.kern = kern
        self.likelihood = likelihood

        if isinstance(X, np.ndarray):
            # X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            # Y is a data matrix, rows correspond to the rows in X,
            # columns are treated independently
            Y = DataHolder(Y)
        self.X, self.Y = X, Y

    @autoflow()
    def compute_log_likelihood_with_gradients(self):
        gradient = tf.gradients(self.likelihood_tensor, self.trainable_tensors)
        return self.likelihood_tensor, gradient

    @autoflow((settings.float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self._build_predict(Xnew)

    @autoflow((settings.float_type, [None, None]))
    def predict_x(self, Xnew):
        """
        Compute the mean and variance of the latent x(s) at the points
        Xnew.
        """
        return self._build_predict_x(Xnew)

    @autoflow((settings.float_type, [None, None]))
    def sample_x(self, Xnew):
        """
        Compute the mean and variance of the latent x(s) at the points
        Xnew.
        """
        return self._build_sample_x(Xnew)

    @autoflow((settings.float_type, [None, None]))
    def test_log_likelihood(self, Xnew):
        """
        Compute the mean and variance of the latent x(s) at the points
        Xnew.
        """
        return self._test_likelihood(Xnew)

    @autoflow((settings.float_type, [None, None]))
    def test_predict_x(self, Xnew):
        """
        Compute the mean and variance of the latent x(s) at the points
        Xnew.
        """
        return self._test_predict_x(Xnew)

    @autoflow()
    def get_uZ(self):
        """
        Compute the mean and variance of the latent x(s) at the points
        Xnew.
        """
        return self.uZ

    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]), (settings.float_type, [None, None]), (settings.float_type, [None, None]))
    def my_base_conditional(self, Kmn, Kmm_sigma, Knn, y):
        """
        Compute the mean and variance of the latent x(s) at the points
        Xnew.
        """
        return self._base_cond(Kmn, Kmm_sigma, Knn, y, full_cov=True, white=False)

    @autoflow((settings.float_type, [None, None]), (settings.int_type, []))
    def jacobian_x(self, Xnew, f_i):
        """
        Compute the jacobian of a multi-output GP w.r.t. Xnew.
        !! WARNING: Assumes the decomposition is [0,1,2], [3,4,5], [6,7,8], ... !!
        !! WARNING: Assumes Xnew has shape 1 x d !!
        """
        # compute MOGP prediction
        # mean_x, _ = self._build_predict_x(Xnew)
        mean_x, _ = self._build_predict_x(Xnew)
        # reshape to obtain M x D tensor
        mean_x_reshape = tf.transpose(tf.reshape(mean_x, [len(self.decomposition), self.Mo_dim, tf.shape(Xnew)[0]]),
                                      perm=[0, 2, 1])
        mu_x = tf.concat([mean_x_reshape[kk, :, :] for kk in list(range(len(self.decomposition)))], axis=1)

        # compute gradients
        def remove_none(grad):  # , xx):
            if grad == None:
                return tf.zeros_like(Xnew, dtype=Xnew.dtype)
                # return tf.zeros_like(xx, dtype=xx.dtype)
            else:
                return grad

        # jac_per_point = []
        # for x_i in range(self.M):
        #     jac_per_point.append(tf.concat(
        #         [remove_none(tf.gradients(mu_x[x_i, f_i], Xnew)[0])[x_i, :][None] for f_i in range(self.input_dim)],
        #         axis=0))
        # jac_per_point = tf.concat([remove_none(tf.gradients(mu_x[0, f_i], Xnew)[0]) for f_i in range(self.input_dim)],
        #                           axis=0)
        jac_per_point = tf.gradients(mu_x[0, f_i], Xnew)[0]
        # jac_per_point.append(remove_none(tf.gradients(mu_x[0, 0], Xnew[0:5, :])[0], Xnew[0:5, :]))
        return mu_x, jac_per_point

    @autoflow((settings.float_type, [None, None]), (settings.int_type, []))
    def jacobian_Fullx(self, Xnew, f_i):
        """
        Compute the jacobian of a multi-output GP w.r.t. Xnew.
        !! WARNING: Assumes Xnew has shape 1 x d !!
        """
        # compute MOGP prediction (only posterior mean)
        mean_x, _ = self._build_predict_x(Xnew)
        # reshape to obtain M x D tensor
        mu_x = tf.transpose(tf.reshape(mean_x, [self.Mo_dim, tf.shape(Xnew)[0]]))

        # compute gradients
        def remove_none(grad):  # , xx):
            if grad == None:
                return tf.zeros_like(Xnew, dtype=Xnew.dtype)
                # return tf.zeros_like(xx, dtype=xx.dtype)
            else:
                return grad

        jac_per_point = tf.gradients(mu_x[0, f_i], Xnew)[0]
        # jac_per_point.append(remove_none(tf.gradients(mu_x[0, 0], Xnew[0:5, :])[0], Xnew[0:5, :]))
        return mu_x, jac_per_point

    @autoflow((settings.float_type, [None, None]), (settings.int_type, []))
    def jacobian_BLRx(self, Xnew, f_i):
        """
        Compute the jacobian of a multi-output GP w.r.t. Xnew.
        !! WARNING: Assumes Xnew has shape 1 x d !!
        """
        # compute MOGP prediction (only posterior mean)
        mean_x, _, _ = self._build_predict_x(Xnew)
        # reshape to obtain M x D tensor
        mu_x = tf.transpose(tf.reshape(tf.transpose(mean_x), [self.Mo_dim, tf.shape(Xnew)[0]]))

        jac_per_point = tf.gradients(mu_x[0, f_i], Xnew)[0]
        return mu_x, jac_per_point

    @autoflow((settings.float_type, [None, None]), (settings.float_type, []), (settings.float_type, []))
    def EI_grad(self, Xnew, ymin, threshold):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        fmean, fvar = self._build_predict(Xnew)
        ymin_th = ymin - threshold

        fstd = tf.sqrt(fvar + 1e-09)
        gauss_distributions = tf.distributions.Normal(loc=fmean[:, 0],
                                                      scale=fstd[:, 0])  # all distributions, scale=stddev

        cdf_vec = gauss_distributions.cdf( ymin_th * tf.ones_like(fmean[:, 0]))[:, None]
        pdf_vec = gauss_distributions.prob(ymin_th * tf.ones_like(fmean[:, 0]))[:, None]

        EI = tf.multiply((ymin_th - fmean), cdf_vec) + tf.multiply(fvar, pdf_vec)  # check shapes
        bool_positive = tf.greater(EI, tf.zeros_like(EI))
        ei = tf.where(bool_positive, EI, tf.zeros_like(EI))
        ei_sum = tf.reduce_sum(tf.negative(ei), axis=0, keepdims=True)
        ei_grad = tf.gradients(ei_sum, Xnew)
        # return tf.negative(ei)
        # return fmean, fvar
        # return cdf_vec, pdf_vec
        # return EI
        # return ei_sum
        return tf.negative(ei), ei_sum, ei_grad  # tested

    @autoflow((settings.float_type, [None, None]), (settings.float_type, []), (settings.float_type, []))
    def PI_grad(self, Xnew, ymin, threshold):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        fmean, fvar = self._build_predict(Xnew)
        ymin_th = ymin - threshold
        fstd = tf.sqrt(fvar + 1e-09)
        gauss_distributions = tf.distributions.Normal(loc=fmean[:, 0], scale=fstd[:, 0])
        pi = gauss_distributions.cdf(ymin_th * tf.ones_like(fmean[:, 0]))[:, None]
        pi_sum = tf.reduce_sum(tf.negative(pi), axis=0, keepdims=True)
        pi_grad = tf.gradients(pi_sum, Xnew)
        return tf.negative(pi), pi_sum, pi_grad

    @autoflow((settings.float_type, [None, None]))
    def LCB_grad(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        fmean, fvar = self._build_predict(Xnew)
        beta = np.sqrt(3.).astype(np.float64)
        fstd = tf.sqrt(fvar + 1e-09)
        lcb_sum = tf.reduce_sum(fmean - tf.multiply(beta, fstd), axis=0, keep_dims=True)
        lcb_grad = tf.gradients(lcb_sum, Xnew)
        return fmean - tf.multiply(beta, fstd), lcb_sum, lcb_grad


    @autoflow((settings.float_type, [None, None]), (settings.float_type, []), (settings.float_type, []), (settings.float_type, []))
    def EI_grad_mo(self, Xnew, ymin, threshold, input_dim):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        fmean, fvar = self._build_predict(tf.concat([Xnew, tf.ones(shape=[tf.shape(Xnew)[0], 1], dtype=settings.float_type) * input_dim], axis=1))
        ymin_th = ymin - threshold

        fstd = tf.sqrt(fvar + 1e-09)
        gauss_distributions = tf.distributions.Normal(loc=fmean[:, 0],
                                                      scale=fstd[:, 0])  # all distributions, scale=stddev

        cdf_vec = gauss_distributions.cdf( ymin_th * tf.ones_like(fmean[:, 0]))[:, None]
        pdf_vec = gauss_distributions.prob(ymin_th * tf.ones_like(fmean[:, 0]))[:, None]

        EI = tf.multiply((ymin_th - fmean), cdf_vec) + tf.multiply(fvar, pdf_vec)  # check shapes
        bool_positive = tf.greater(EI, tf.zeros_like(EI))
        ei = tf.where(bool_positive, EI, tf.zeros_like(EI))
        ei_sum = tf.reduce_sum(tf.negative(ei), axis=0, keepdims=True)
        ei_grad = tf.gradients(ei_sum, Xnew)
        # return tf.negative(ei)
        # return fmean, fvar
        # return cdf_vec, pdf_vec
        # return EI
        # return ei_sum
        return tf.negative(ei), ei_sum, ei_grad  # tested

    @autoflow((settings.float_type, [None, None]), (settings.float_type, []), (settings.float_type, []), (settings.float_type, []))
    def PI_grad_mo(self, Xnew, ymin, threshold, input_dim):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        fmean, fvar = self._build_predict(tf.concat([Xnew, tf.ones(shape=[tf.shape(Xnew)[0], 1], dtype=settings.float_type) * input_dim], axis=1))
        ymin_th = ymin - threshold
        fstd = tf.sqrt(fvar + 1e-09)
        gauss_distributions = tf.distributions.Normal(loc=fmean[:, 0], scale=fstd[:, 0])
        pi = gauss_distributions.cdf(ymin_th * tf.ones_like(fmean[:, 0]))[:, None]
        pi_sum = tf.reduce_sum(tf.negative(pi), axis=0, keepdims=True)
        pi_grad = tf.gradients(pi_sum, Xnew)
        return tf.negative(pi), pi_sum, pi_grad

    @autoflow((settings.float_type, [None, None]), (settings.float_type, []))
    def LCB_grad_mo(self, Xnew, input_dim):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        fmean, fvar = self._build_predict(tf.concat([Xnew, tf.ones(shape=[tf.shape(Xnew)[0], 1], dtype=settings.float_type) * input_dim], axis=1))
        beta = np.sqrt(3.).astype(np.float64)
        fstd = tf.sqrt(fvar + 1e-09)
        lcb_sum = tf.reduce_sum(fmean - tf.multiply(beta, fstd), axis=0, keep_dims=True)
        lcb_grad = tf.gradients(lcb_sum, Xnew)
        return fmean - tf.multiply(beta, fstd), lcb_sum, lcb_grad

    # @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    # def KLconstraint(self, x, Xnn):
    #     # x   assumed of shape M x d
    #     # Xnn assumed of shape N x d
    #     xs = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)        # M x 1
    #     Xs = tf.reduce_sum(tf.square(Xnn), axis=-1, keepdims=True)      # N x 1
    #     xXT = tf.matmul(x, Xnn, transpose_b=True)                       # M x N
    #     KL = (xs + tf.transpose(Xs) - 2. * xXT) / 2.                    # M x N
    #     ind_minimizers = tf.math.argmin(KL, axis=1, output_type=tf.int64)   # M x 1 or (M,) ? check that, (not relevant)
    #     zi_opts = tf.gather(Xnn, tf.squeeze(ind_minimizers), axis=0)        # closest latent training points to each (M x d)
    #     mean_zi, var_zi = self._build_predict_x(zi_opts)
    #
    #     # KL_min = tf.reduce_min(KL, axis=0, keepdims=False)
    #     return mean_zi

    @autoflow((settings.float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self._build_predict(Xnew, full_cov=True)

    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self._build_predict(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=settings.float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    @autoflow((settings.float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

    @abc.abstractmethod
    def _build_predict(self, *args, **kwargs):
        raise NotImplementedError('') # TODO(@awav): write error message