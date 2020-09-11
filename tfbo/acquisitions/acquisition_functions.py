import numpy as np
from scipy.stats import norm


def Neg_ei(x, gpmodel, ymin, threshold=1e-04):
    fmean, fvar = gpmodel.predict_f(x)  # check shape
    fvar = np.maximum(1e-09, fvar)
    ymin_th = ymin - threshold
    fstd = np.sqrt(fvar + 1e-09)
    Z = (ymin_th - fmean) / fstd
    cdf = norm.cdf(ymin_th * np.ones_like(fmean), loc=fmean, scale=fstd)
    pdf = norm.pdf(Z) / fstd
    expected_improvement = np.multiply(ymin_th-fmean, cdf) + np.multiply(pdf, fvar)
    expected_improvement_corr = np.where(expected_improvement < 0.,
                                         np.zeros_like(expected_improvement), expected_improvement)
    return - expected_improvement_corr  # check shape
    # return cdf, pdf
    # return expected_improvement


def Neg_pi(x, gpmodel, ymin, threshold=1e-04):
    fmean, fvar = gpmodel.predict_f(x)
    fvar = np.maximum(1e-09, fvar)
    fstd = np.sqrt(fvar + 1e-09)
    ymin_th = ymin - threshold
    cdf = norm.cdf(ymin_th * np.ones_like(fmean), loc=fmean, scale=fstd)
    return - cdf    # check shape


def lcb(x, gpmodel):
    fmean, fvar = gpmodel.predict_f(x)
    fvar = np.maximum(1e-09, fvar)
    fstd = np.sqrt(fvar + 1e-09)
    beta = np.sqrt(3.)
    lcb = fmean - beta * fstd
    return lcb  # check shape


def Neg_ei_wg(x, gpmodel, ymin, threshold=1e-04):
    ei_arr, ei_sum, ei_grad = gpmodel.EI_grad(x, ymin, threshold)    # make sure ymin is just number
    return ei_arr, ei_sum, ei_grad  # check shape


def Neg_pi_wg(x, gpmodel, ymin, threshold=1e-04):
    pi_arr, pi_sum, pi_grad = gpmodel.PI_grad(x, ymin, threshold)    # make sure ymin is just number
    return pi_arr, pi_sum, pi_grad  # check shape


def lcb_wg(x, gpmodel):
    lcb_arr, lcb_sum, lcb_grad = gpmodel.LCB_grad(x)
    return lcb_arr, lcb_sum, lcb_grad   # check shape



def Neg_ei_mo(x, gpmodel, ymin, input_dim, threshold=1e-04):
    ei_arr, ei_sum, ei_grad = gpmodel.EI_grad_mo(x, ymin, threshold, input_dim)    # make sure ymin is just number
    return ei_arr, ei_sum, ei_grad  # check shape


def Neg_pi_mo(x, gpmodel, ymin, input_dim, threshold=1e-04):
    pi_arr, pi_sum, pi_grad = gpmodel.PI_grad_mo(x, ymin, threshold, input_dim)    # make sure ymin is just number
    return pi_arr, pi_sum, pi_grad  # check shape


def lcb_mo(x, gpmodel, input_dim):
    lcb_arr, lcb_sum, lcb_grad = gpmodel.LCB_grad_mo(x, input_dim)
    return lcb_arr, lcb_sum, lcb_grad   # check shape


def Neg_ei_dgp(x, gpmodel, ymin, threshold=1e-04, num_samples=1):
    ei_arr, ei_sum, ei_grad = gpmodel.EI_grad(x, ymin, threshold, num_samples)    # make sure ymin is just number
    return ei_arr, ei_sum, ei_grad  # check shape


def Neg_pi_dgp(x, gpmodel, ymin, threshold=1e-04, num_samples=1):
    pi_arr, pi_sum, pi_grad = gpmodel.PI_grad(x, ymin, threshold, num_samples)    # make sure ymin is just number
    return pi_arr, pi_sum, pi_grad  # check shape


def lcb_dgp(x, gpmodel, num_samples=1):
    lcb_arr, lcb_sum, lcb_grad = gpmodel.LCB_grad(x, num_samples)
    return lcb_arr, lcb_sum, lcb_grad   # check shape