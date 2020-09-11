import numpy as np
from tfbo.acquisitions.acquisition_functions import Neg_ei, Neg_pi, lcb


# test output negative_ei
x = np.linspace(start=0., stop=2*np.pi, num=500)[:, None]
xtrain = np.linspace(start=0., stop=2*np.pi, num=50)[:, None]
ytrain = np.sin(xtrain)
import gpflow
kernel = gpflow.kernels.Matern52(input_dim=1, ARD=True)
gpmodel = gpflow.models.GPR(X=xtrain, Y=ytrain, kern=kernel)
ymin = ytrain.min()
neg_ei = Neg_ei(x, gpmodel, ymin)
import tensorflow as tf
def neg_expected_improvement(post_mean, post_cov, ymin=None, threshold=1e-04, **kwargs):    # tested
    '''
    Definition of Expected improvement assuming a single-pool acquisition function evaluation and assuming we want to minimize the objective
    :param post_mean: GP-mean predictionas  Ntest x 1
    :param post_cov: GP-var_predictions     Ntest x 1
    :param ymin: best observation
    :param threshold: minimum improvement
    :return: N x 1
    '''
    ymin_th = ymin - threshold
    post_scales = tf.sqrt(post_cov + 1e-09)
    gauss_distributions = tf.distributions.Normal(loc=post_mean[:, 0], scale=post_scales[:, 0])  # all distributions, scale=stddev
    cdf_vec = gauss_distributions.cdf(ymin_th*tf.ones_like(post_mean)[:, 0])[:, None]
    pdf_vec = gauss_distributions.prob(ymin_th*tf.ones_like(post_mean[:, 0]))[:, None]
    # post_mean_list = tf.unstack(post_mean, axis=0)
    # post_cov_list = tf.unstack(post_cov, axis=0)
    # destrib_list = [tf.distributions.Normal(loc=post_mean_i, scale=post_cov_i) for post_mean_i, post_cov_i in zip(post_mean_list, post_cov_list)]
    # cdf_vec = tf.concat([distrib_i.cdf(ymin) for distrib_i in destrib_list], axis=0)
    # pdf_vec = tf.concat([distrib_i.pdf(ymin) for distrib_i in destrib_list], axis=0)
    EI = tf.multiply((ymin_th-post_mean), cdf_vec) + tf.multiply(post_cov, pdf_vec)        # check shapes
    bool_positive = tf.greater(EI, tf.zeros_like(EI))
    ei = tf.where(bool_positive, EI, tf.zeros_like(EI))
    return tf.negative(ei)
fmean, fvar = gpmodel.predict_f(x)
fvar = np.maximum(1e-09, fvar)
neg_ei_tf = neg_expected_improvement(fmean, fvar, ymin)
sess = tf.InteractiveSession()
_neg_ei = sess.run(neg_ei_tf)

err_neg_ei = np.max(np.abs(_neg_ei-neg_ei))


# test output pi
x = np.linspace(start=0., stop=2*np.pi, num=500)[:, None]
xtrain = np.linspace(start=0., stop=2*np.pi, num=50)[:, None]
ytrain = np.sin(xtrain)
ymin = ytrain.min()
import gpflow
kernel = gpflow.kernels.Matern52(input_dim=1, ARD=True)
gpmodel = gpflow.models.GPR(X=xtrain, Y=ytrain, kern=kernel)
neg_pi = Neg_pi(x=x, gpmodel=gpmodel, ymin=ymin)
import tensorflow as tf
def neg_probability_of_improvement(post_mean, post_cov, ymin=None, threshold=1e-04, **kwargs):  #tested
    '''
    Probability of improvement acquisition function
    :param post_mean: Ntest x 1
    :param post_cov: Ntest x 1
    :param ymin: Ntest x 1
    :param threshold: minimum improvement
    :return: Ntes x 1
    '''
    ymin_th = ymin - threshold
    post_scales = tf.sqrt(post_cov + 1e-09)
    gauss_distributions = tf.distributions.Normal(loc=post_mean[:, 0], scale=post_scales[:, 0])
    pi = gauss_distributions.cdf(ymin_th*tf.ones_like(post_mean[:, 0]))[:, None]
    return tf.negative(pi)
fmean, fvar = gpmodel.predict_f(x)
fvar = np.maximum(1e-09, fvar)
neg_pi_tf = neg_probability_of_improvement(fmean, fvar, ymin)
sess = tf.InteractiveSession()
_neg_pi = sess.run(neg_pi_tf)

err_neg_pi = np.max(np.abs(_neg_pi - neg_pi))


x = np.linspace(start=0., stop=2*np.pi, num=500)[:, None]
xtrain = np.linspace(start=0., stop=2*np.pi, num=50)[:, None]
ytrain = np.sin(xtrain)
import gpflow
kernel = gpflow.kernels.Matern52(input_dim=1, ARD=True)
gpmodel = gpflow.models.GPR(X=xtrain, Y=ytrain, kern=kernel)
lcb_out = lcb(x, gpmodel)
import tensorflow as tf
def lc_bound(post_mean, post_cov, **kwargs):    # tested
    '''
    Lower confidence bound acquisition function
    :param post_mean: Ntest x 1
    :param post_cov: Ntest x 1
    :return: Ntest x 1
    '''
    beta = np.sqrt(3.).astype(np.float64)
    stddev_post = tf.sqrt(post_cov + 1e-09)
    return post_mean - tf.multiply(beta, stddev_post)
fmean, fvar = gpmodel.predict_f(x)
fvar = np.maximum(1e-09, fvar)
lcb_tf = lc_bound(fmean, fvar)
sess = tf.InteractiveSession()
_lcb = sess.run(lcb_tf)

err_lcb = np.max(np.abs(_lcb - lcb_out))