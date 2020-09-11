from tfbo.models.gp_models import GPModel
from gpflow import models, likelihoods
from gpflow.params import DataHolder
import tensorflow as tf
from gpflow import settings
from gpflow import logdensities
from gpflow import params_as_tensors
stability = settings.jitter
import numpy as np


class AddGPR(GPModel):
    '''
    Additive Gaussian process model. The model takes an additional input "i" that contains a list of integers.
    Each index i is associated to a kernel which acts on a specific set of active dimensions
    '''
    def __init__(self, X, Y, kern, mean_function=None, i=None, **kwargs):
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.i = i

    @params_as_tensors
    def _build_likelihood(self):     # mean_function !
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        jitter = self.stable_jitter(K)
        L = tf.cholesky(K + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * jitter)
        m = self.mean_function(self.X)
        # (R,) log-likelihoods for each independent dimension of Y
        logpdf = logdensities.multivariate_normal(self.Y, m, L)
        return tf.reduce_sum(logpdf)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        Kx = self.kern.K(self.X, Xnew, i=self.i)
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance

        determinant = tf.matrix_determinant(K)
        K = tf.Print(K, ["Determinant of K matrix: ",  determinant])

        jitter = self.stable_jitter(K)
        L = tf.cholesky(K + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * jitter)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew, i=self.i) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew, i=self.i) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar
        # return Kx

    def stable_jitter(self, K):
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


class GPR_stable(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
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
        logpdf = logdensities.multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        return tf.reduce_sum(logpdf)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        jitter = self.stable_jitter(K)                                  # either smallest eigenvalue plus jitter or zero
        L = tf.cholesky(K + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * jitter)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar
        # return jitter * tf.ones_like(fmean), jitter * tf.ones_like(fmean)
        # return self.likelihood.variance

    def stable_jitter(self, K):
        determinant = tf.matrix_determinant(K)
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


class QGPR_stable(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """

    def __init__(self, X, Y, kern, quantile, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.quantile = quantile
        self.ttau = None
        self.tnu = None

    @params_as_tensors
    def _build_likelihood(self):
        K = self.kern.K(self.X)
        ttau_out, tnu_out = self.compute_site_parameters()
        llik = self.compute_log_marginal_lik(ttau=ttau_out, tnu=tnu_out, K=K)
        self.ttau = ttau_out
        self.tnu = tnu_out
        # return tf.negative(llik)
        return llik
        # return ttau_out, tnu_out

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        K = self.kern.K(self.X)
        # ttau, tnu = self.compute_site_parameters()     # data already normalized
        ttau = self.ttau
        tnu = self.tnu      # the model needs to be trained first!
        ts = tf.sqrt(tf.abs(ttau))
        B = tf.eye(tf.shape(ttau)[0], dtype=settings.float_type, name='Ipost') + \
            tf.multiply(tf.matmul(ts, ts, transpose_b=True), K)
        L = tf.cholesky(B)

        KXx = self.kern.K(self.X, Xnew)
        C = tf.matrix_triangular_solve(L, tf.multiply(ts, KXx))

        G = tf.matrix_triangular_solve(L, tf.diag(ts[:, 0]))
        inner_mat = tf.eye(tf.shape(ttau)[0], dtype=settings.float_type) - \
                    tf.matmul(tf.matmul(G, G, transpose_a=True), K)
        SmX = tf.multiply(ttau, self.mean_function(self.X))
        fmean = self.mean_function(Xnew) + tf.matmul(KXx, tf.matmul(inner_mat, tnu - SmX), transpose_a=True)
        # if correction_term:
        #     if self.mean == zero_mean:  raise ValueError('Correction term requires parametrized mean function')
        #     else:
        #         alpha = self.select_hyps(hyp_def='mean_hyp')
        #         m = self.mean_fn(X)
        #         mstar = self.mean_fn(Xtest)
        #         g_func_m = lambda i: tf.gradients(m[:, 0], alpha, grad_ys=tf.one_hot(i, depth=tf.shape(X)[0], dtype=tf.float64))
        #         g_func_mstar = lambda i:tf.gradients(mstar[:, 0], alpha, grad_ys=tf.one_hot(i, depth=tf.shape(Xtest)[0], dtype=tf.float64))
        #         dm_dalpha_tf = tf.map_fn(g_func_m, elems=tf.range(start=0, limit=tf.shape(X)[0], delta=1), dtype=[tf.float64]*len(alpha))
        #         dmstar_dalpha_tf = tf.map_fn(g_func_mstar, elems=tf.range(start=0, limit=tf.shape(Xtest)[0], delta=1), dtype=[tf.float64]*len(alpha))
        #         dm_dalpha = tf.stack(dm_dalpha_tf, axis=1)
        #         dmstar_dalpha = tf.stack(dmstar_dalpha_tf, axis=1)
        #     # if self.mean == constant_mean:    # hard-coded derivatives given mean function
        #     #     dm_dalpha = tf.ones(shape=[tf.shape(X)[0], 1], dtype=tf.float64)
        #     #     dmstar_dalpha = tf.ones(shape=[tf.shape(Xtest)[0], 1], dtype=tf.float64)
        #     # elif self.mean == linear_mean:
        #     #     dm_dalpha = tf.concat(values=[X, tf.ones(shape=[tf.shape(X)[0], 1], dtype=tf.float64)], axis=1)   # concat_dim=1,              # Nxm, N=num_input_train
        #     #     dmstar_dalpha = tf.concat(values=[Xtest, tf.ones(shape=[tf.shape(Xtest)[0], 1], dtype=tf.float64)] , axis=1)   # concat_dim=1,     # nxm, n=num_test_inputs, m=num_mean_parameters (self.input_dim+1)
        #     # else: raise ValueError('mean_function not implemented for computation of correction term')
        #     S_tilde = tf.diag(ttau[:, 0])
        #     K_Sinv_inv = tf.matmul(inner_mat, S_tilde)
        #     w = tf.transpose(tf.matmul(KXx, K_Sinv_inv, transpose_a=True))   # K_starT(K+S_tilde^(-1))^(-1), Nxn
        #     g = dmstar_dalpha - tf.transpose(tf.matmul(dm_dalpha, w, transpose_a=True))           # nxm
        #     M = tf.matmul(dm_dalpha, tf.matmul(K_Sinv_inv, dm_dalpha), transpose_a=True)            # mxm
        #     LM = tf.cholesky(M + 1e-06*tf.eye(self.num_mean_hyps, dtype=tf.float64))
        #     J = tf.matrix_triangular_solve(LM, tf.transpose(g))
        #     Ecorrection = tf.diag_part(tf.matmul(J, J, transpose_a=True))[:, None]                           # nx1
        #     post_cov = post_cov + tf.diag(Ecorrection[:, 0])
        #     # return dm_dalpha_stack, dmstar_dalpha_stack, dm_dalpha, dmstar_dalpha
        #     # return dmstar_dalpha, dm_dalpha, K, KXx, w, ttau, M, g, Ecorrection, tf.diag_part(post_cov)   # for test
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(C, C, transpose_a=True)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.diag_part(tf.matmul(C, C, transpose_a=True))
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

    def likALD(self, y, S2, mu, derivatives=False):
        '''
        This returns the log of the integral that is log(Z) = "lZ" = int(p(y|f)N(f;mu, S2)df)
        p(y|f): Asymmetric laplace likelihood ALD(y)
        N(f;mu, S2): distribution for the expectation w.r.t. f
        See The GPML Toolbox version 3.2, page 13
        :param y:   must be Nx1
        :param S2:  must be Nx1
        :param mu:  must be Nx1
        :param sn:  must be Nx1 likelihood std-dev
        :param quantile:    just a value
        :param derivatives:    boolean to obtain higher order derivatives
        :return: log(int(p(y|f)N(f;mu, S2)df)), d/dmu log(int(p(y|f)N(f;mu, S2)df)), d^2/dmu^2 log(int(p(y|f)N(f;mu, S2)df))
        '''
        def logphi(z):
            lp = tf.zeros_like(z)
            zmin = tf.constant(-6.2, dtype=settings.float_type, shape=[], name='max-z-value')
            zmax = tf.constant(-5.5, dtype=settings.float_type, shape=[], name='min-z-value')
            oks = tf.greater(z, zmax)
            bds = tf.less(z, zmin)
            ip = tf.logical_and(tf.logical_not(oks), tf.logical_not(bds))
            z_ip = tf.where(ip, z, tf.zeros_like(z))
            lam = 1. / (1. + tf.exp(25. * (0.5 - tf.divide(z_ip - zmin, zmax - zmin))))
            z_ok = tf.where(oks, z, tf.zeros_like(z))
            lp_ok = tf.log(tf.abs((1. + tf.erf(z_ok / np.sqrt(2.))) / 2.))
            z_not_ok = tf.where(tf.logical_not(oks), z, tf.zeros_like(z))
            lp_not_ok = -np.log(np.pi) / 2. - (z_not_ok ** 2) / 2. - tf.log(
                tf.abs(tf.sqrt((z_not_ok ** 2) / 2. + 2.) - z_not_ok / np.sqrt(2.)))
            lp_wok = tf.where(oks, lp_ok, lp)  # with modification only on oks=True
            lp_wok_wnok = tf.where(tf.logical_not(oks), lp_not_ok, lp_wok)  # with modification only on oks=False

            lp_ip = tf.multiply(1. - lam, lp_wok_wnok) + tf.multiply(lam, tf.log(
                tf.abs((1. + tf.erf(z_ip / np.sqrt(2.))) / 2.)))
            lp_return = tf.where(ip, lp_ip, lp_wok_wnok)  # with modification only on oks=True
            return lp_return

        def logsum2exp(logx):
            ones = tf.ones_like(logx)[0, :][None]
            max_logx = tf.reduce_max(logx, axis=1, keepdims=True, name='maximum-over-columns')
            x = tf.exp(logx - tf.matmul(max_logx, ones))
            y = tf.log(tf.abs(tf.reduce_sum(x, axis=1, keepdims=True))) + max_logx
            return y

        def expABz_expAx(A, x, B, z):
            ones = tf.ones_like(A)[0, :][None]
            maxA = tf.reduce_max(A, axis=1, keepdims=True)
            A = A - tf.matmul(maxA, ones)
            y = tf.divide(tf.matmul(tf.multiply(tf.exp(A), B), z), tf.matmul(tf.exp(A), x))
            return y

        sn = tf.sqrt(tf.maximum(self.likelihood.variance, 1e-40))
        quantile = self.quantile
        # if (sn is None): sn = tf.exp(0.5 * self.state['log_noise_var'])   # likelihood std-dev hyper-parameter
        # if (quantile is None): quantile = self.quantile     # set quantile q

        # sn = tf.exp(0.5 * self.state['log_noise_var'])
        # sn = tf.maximum(tf.constant(1e-06, dtype=settings.float_type, shape=[], name='min-sn-value' + '_gp' + str(self.num_gp)), sn)

        tmu = (mu - y) / sn
        tvar = S2 / self.likelihood.variance

        tstd = tf.sqrt(tf.maximum(tvar, 1e-40))     # tf.sqrt(tf.abs(tvar))
        tstd = tf.maximum(tf.constant(1e-06, dtype=settings.float_type, shape=[], name='min-std-dev-value'), tstd)  # maximum supports broadcasting
        zm = tmu / tstd + quantile * tstd
        zp = zm - tstd

        Lm = tmu * quantile + 0.5 * tvar * quantile**2

        am = logphi(-zm)
        ap = logphi(zp) - tmu - tvar * (quantile - 0.5)

        lZ = logsum2exp(tf.concat([ap, am], axis=1)) + Lm - tf.log(tf.abs(sn)) + \
             tf.cast(tf.log(tf.abs(quantile * (1 - quantile))), dtype=settings.float_type)
        # tested up to here with matlab implementation. Continue with higher order derivatives.
        if (derivatives == True):
            # First order derivative: d/dmu (log(Z))
            lqm = -0.5 * (zm**2) - 0.5 * tf.cast(tf.log(2.*np.pi), dtype=settings.float_type) - am
            lqp = -0.5 * (zp**2) - 0.5 * tf.cast(tf.log(2.*np.pi), dtype=settings.float_type) - logphi(zp)

            dam = -tf.exp(lqm - 0.5 * tf.log(tf.abs(S2)))
            dap =  tf.exp(lqp - 0.5 * tf.log(tf.abs(S2))) - 1. / sn

            dLm = quantile / sn

            dlZ = expABz_expAx(tf.concat([ap, am], axis=1),   tf.ones(shape=[2, 1], dtype=settings.float_type),
                               tf.concat([dap, dam], axis=1), tf.ones(shape=[2, 1], dtype=settings.float_type)) + dLm
            # tested up to first derivative with matlab implementation

            # Second order derivative: d^2/dmu^2 (log(Z))   # implementation of second derivative only for 1x1 inputs
            bm = tf.matmul(tf.divide(-zm, tf.transpose(tf.sqrt(tf.abs(S2)))), dam)    # NxN * Nx1 = Nx1
            bp = - tf.multiply(tf.divide(zp, tf.transpose(tf.sqrt(tf.abs(S2)))), dap+1./sn) - 2.0 * tf.divide(dap, sn) \
                 - 1./(sn**2)     # NxN * Nx1 = Nx1

            d2lZ = expABz_expAx(tf.concat([ap, am], axis=1), tf.ones(shape=[2, 1], dtype=settings.float_type),
                                tf.concat([bp, bm], axis=1), tf.ones(shape=[2, 1], dtype=settings.float_type)) - \
                   (dlZ - dLm)**2
            # tested assuming 1x1 shapes for: y, S2, mu, sn
            return lZ, dlZ, d2lZ
        return lZ


    def compute_site_parameters(self):    # [!] this function is assumed to be called by other methods of the class, IT does not normalize data [!]
        '''
        Compute EP approximation parameters
        1) Initialize site parameters: ttau, tnu, tZ, posterior mean and covariance
        2) Iteratively update all site parameters
        2.1) Iteratively update each site parameter
        2.2) Iteratively update posterior mean anc covariance
        3) Recompute posterior mean and covariance to keep numerical precision
        :return:
        '''
        # 1)
        ttau = tf.zeros_like(self.Y)
        tnu = tf.zeros_like(self.Y)
        K = self.kern.K(self.X)
        Sigma = self.kern.K(self.X)
        mu = self.mean_function(self.X)     # tf.zeros_like(y)   # self.mean(X)
        sweep = tf.constant(0, dtype=settings.int_type, shape=[], name='counter-outer-loop')

        def not_converged(y, ttau, tnu, K, Sigma, mu, sweep):
            # min_sweep = tf.constant(2, dtype=settings.int_type, shape=[], name='min-counter-outer-loop')
            # max_sweep = tf.constant(10, dtype=settings.int_type, shape=[], name='max-number-outer-loops')
            # tol = tf.constant(1e-04, dtype=settings.float_type, shape=[], name='minimum-tolerance-change-nlZ')
            # return tf.logical_or(
            #     tf.logical_and(tf.less(sweep, max_sweep), tf.less(tol, tf.abs(nlZ - nlZ_old))),
            #     tf.less(sweep, min_sweep))
            return tf.less(sweep, tf.constant(10, dtype=settings.int_type, shape=[], name='max-number-outer-loops'))

        def outer_loop(y, ttau, tnu, K, Sigma, mu, sweep):
            i = tf.constant(0, dtype=settings.int_type, shape=[], name='inner-iteration-variable')    # inner loop index
            y_end, ttau_end, tnu_end, Sigma_end, mu_end, i_end = \
                tf.while_loop(not_all_params, inner_loop, (y, ttau, tnu, Sigma, mu, i))
            Sigma_update, mu_update = self.update_posterior_params(ttau_end, tnu_end, K)
            return y, ttau_end, tnu_end, K, Sigma_update, mu_update, sweep + 1 # y,K unchanged but necessary for inner loop and update posterior

        def not_all_params(y, ttau, tnu, Sigma, mu, i):
            return tf.less(i, tf.shape(y)[0])

        def inner_loop(y, ttau, tnu, Sigma, mu, i):
            ttau_i, tnu_i, Sigma_i, mu_i = self.update_single_site_param(y, ttau, tnu, Sigma, mu, i)
            return y, ttau_i, tnu_i, Sigma_i, mu_i, i + 1

        y_out, ttau_out, tnu_out, K_out, Sigma_out, mu_out, sweep_out = \
            tf.while_loop(not_converged, outer_loop, (self.Y, ttau, tnu, K, Sigma, mu, sweep))
        ttau_out = tf.stop_gradient(ttau_out)
        tnu_out = tf.stop_gradient(tnu_out)
        return ttau_out, tnu_out    # tested


    def update_posterior_params(self, ttau_i, tnu_i, K):
        '''
        Recompute posterior parameters after each update of all site parameters ttau_i, tnu_i, tZ_i
        :param ttau_i: Nx1
        :param tnu_i: Nx1
        :param K: Nx1
        :return:
        '''
        ts = tf.sqrt(tf.abs(ttau_i))    # Nx1
        B = tf.eye(tf.shape(ttau_i)[0], dtype=settings.float_type) + tf.multiply(tf.matmul(ts, ts, transpose_b=True), K)
        L = tf.cholesky(B)  # + tf.eye(tf.shape(ttau_i)[0], dtype=settings.float_type)
        V = tf.matrix_triangular_solve(L, tf.multiply(ts, K))   # check broadcast
        new_Sigma = K - tf.matmul(V, V, transpose_a=True)
        new_mu = tf.matmul(new_Sigma, tnu_i)
        new_mu = tf.reshape(new_mu, tf.shape(self.mean_function(self.X)))
        # return B, L, V
        return new_Sigma, new_mu    # tested


    def update_single_site_param(self, y, ttau, tnu, Sigma, mu, i):
        '''
        Update single tilde couple-of-parameters (ttau_i, tnu_i), and update posterior accordingly (Sigma_new, mu_new)
        1) Select single tilde_parameters
        2) Compute derivative w.r.t. cavity mean of the expectation: int(p(y|f)p(f)df) w.r.t. cavity distribution p(f)
        3) Update ttau_i, tnu_i with the second and frist derivatives
        4) Update posterior parameters with rank1 update and mu update from Sigma_new
        :param y: observations Nx1
        :param ttau: tilde-precisions Nx1
        :param tnu: tilde-natural-mean Nx1
        :param Sigma: posterior covariance NxN
        :param mu: posterior mean Nx1
        :param i: iteration index ()
        :return: 1-updated tilde_parameters and posterior_parameters
        '''
        y_i = y[i, :][None]
        ttau_i_old = ttau[i, :][None]
        tnu_i_old = tnu[i, :][None]
        Sigma_ii = Sigma[i, i][None, None]
        mu_i = mu[i, :][None]
        # Cavity parameters
        tau_ni = 1./Sigma_ii - ttau_i_old
        nu_ni = mu_i/Sigma_ii - tnu_i_old   # cavity parameters do not depend on self.mean_fn(X),
        # Derivatives 0,1,2 w.r.t. cavity mean
        # sn = tf.sqrt(tf.maximum(self.likelihood.variance, 1e-40))  # tf.exp(0.5 * self.state['log_noise_var'])
        lZ, dlZ, d2lZ = self.likALD(y_i, 1./tau_ni, nu_ni/tau_ni, derivatives=True)   # nans from here? tf.div(np.float64(1.), tau_ni)
        # Update single site-param
        new_ttau_i = - tf.divide(d2lZ,  (1. + tf.divide(d2lZ, tau_ni)))
        new_ttau_i = tf.maximum(new_ttau_i, 0.)     # tf.constant(0., dtype=settings.float_type, name='lower-bound-ttau')
        new_tnu_i = tf.divide((dlZ - tf.divide(nu_ni, tau_ni)*d2lZ), (1. + tf.divide(d2lZ, tau_ni)))  # here
        # Substitute new tilde parameters
        i_th_index = tf.equal(tf.range(tf.shape(ttau)[0])[:, None], i*tf.ones_like(ttau, dtype=settings.int_type))    # check shape
        ttau_new = tf.where(i_th_index, new_ttau_i*tf.ones_like(ttau), ttau)
        tnu_new = tf.where(i_th_index, new_tnu_i*tf.ones_like(tnu), tnu)
        # tilted_tau = ttau_new + tau_ni
        # dttau = tilted_tau - tau_ni - ttau_i_old
        # Update posterior parameters Sigma, mu
        dttau = new_ttau_i - ttau_i_old
        dtnu = new_tnu_i - tnu_i_old
        s_i = Sigma[:, i][:, None]
        c_i = dttau/(1.+dttau*s_i[i, :][None])   # here
        Sigma_new = Sigma - c_i * tf.matmul(s_i, s_i, transpose_b=True)
        mu_new = mu - (c_i * (mu_i + s_i[i, :][None] * dtnu) - dtnu) * s_i   #tf.matmul(Sigma_new, mu)
        mu_new = tf.reshape(mu_new, tf.shape(self.mean_function(self.X)))
        # return tau_ni, nu_ni
        # return y_i, 1./tau_ni, nu_ni/tau_ni
        # return lZ, dlZ, d2lZ
        return ttau_new, tnu_new, Sigma_new, mu_new     # tested


    def compute_log_marginal_lik(self, ttau, tnu, K):
        '''
        Following instructions on GPML 2006
        :param y:
        :param ttau:
        :param tnu:
        :param K:
        :return:
        '''
        # compute posterior sigma first
        ts = tf.sqrt(tf.abs(ttau))
        B = tf.eye(tf.shape(ttau)[0], dtype=settings.float_type) + tf.multiply(tf.matmul(ts, ts, transpose_b=True), K)
        L = tf.cholesky(B)
        V = tf.matrix_triangular_solve(L, tf.multiply(ts, K))
        Sigma = K - tf.matmul(V, V, transpose_a=True)   # same as in update_posterior_params
        mu = tf.matmul(Sigma, tnu)
        # fourth and first term (ff)
        tau_cavity = 1./tf.diag_part(Sigma)[:, None] - ttau
        ff1 = 0.5 * tf.reduce_sum(tf.log(tf.abs(1 + tf.divide(ttau, tau_cavity))), axis=0, keepdims=True)    # 1x1
        ff2 = tf.reduce_sum(tf.log(tf.abs(tf.diag_part(L)[:, None])), axis=0, keepdims=True)   # 1x1
        ff = ff1 - ff2

        nu_cavity = tf.multiply(mu, 1./tf.diag_part(Sigma)[:, None]) - tnu
        lZ = self.likALD(self.Y, 1./tau_cavity, tf.divide(nu_cavity, tau_cavity), derivatives=False)   # for +sum(lZ)

        m = self.mean_function(self.X)
        p = tnu - tf.multiply(m, ttau)
        pSp05 = 0.5 * tf.matmul(tf.matmul(p, Sigma, transpose_a=True), p)   # gp-likelihood posterior term

        v = tf.diag_part(Sigma)[:, None]
        vTp205 = 0.5 * tf.matmul(v, p**2, transpose_a=True)

        q = nu_cavity - tf.multiply(m, tau_cavity)
        terml = 0.5 * tf.matmul(q, tf.multiply(tf.multiply(ttau/tau_cavity, q) - 2.*p, v), transpose_a=True)

        llik = ff + tf.reduce_sum(lZ, axis=0, keepdims=True) + pSp05 - vTp205 + terml
        return llik     # tested

    def stable_jitter(self, K):
        determinant = tf.matrix_determinant(K)
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