import numpy as np
import gpflow
from tfbo.utils.import_modules import import_attr
from tfbo.models.cov_funcs import Collection, Kstack, Multiple_k
from tfbo.models.gpr_models import AddGPR
import tensorflow as tf
from tfbo.models.gplvm_models import MGPR, Stable_GPR, NN_MoGPR, NN_FullMoGP, NN_BLRMoGP, alpha


def initialize_models(x, y, input_dim, model, kernel, ARD, decomp=None, quantile=None, **kwargs):
    '''
    Initialize the regression models in gpflow, for optimization
    :param xy: dataset
    :param input_dim:
    :param kernel:
    :param ARD:
    :return:
    '''
    if model == 'GPR':
        # with gpflow.defer_build():
        kernel_attr = import_attr('gpflow/kernels', attribute=kernel)
        kernel_out = kernel_attr(input_dim=input_dim, ARD=ARD)
        from tfbo.models.gpr_models import GPR_stable
        gpmodel = GPR_stable(X=x, Y=y, kern=kernel_out)
        # gpmodel.kern.lengthscales.prior = gpflow.priors.Gamma(shape=1., scale=1.)
        # gpmodel.compile()
        # # gpmodel = gpflow.models.GPR(X=x, Y=y, kern=kernel_out)
    if model == 'AddGPR':
        # with gpflow.defer_build():
        # use decomp: list of rank-1 arrays containing active dims
        kernel_attr = import_attr('gpflow/kernels', attribute=kernel)
        def kern_i(kern_attr, input_dim, ARD, decomp_i):
            return kern_attr(input_dim=input_dim, ARD=ARD, active_dims=list(decomp_i))
        kernel_i = lambda d_i: kern_i(kern_attr=kernel_attr, input_dim=input_dim, ARD=ARD, decomp_i=d_i)
        kernels = list(map(kernel_i, decomp))

        k_collection = Collection(kernels)

        def gpmodel_i(x, y, k_collection, indices_kernels):
            return AddGPR(X=x, Y=y, kern=k_collection, i=[indices_kernels])   # indices list for sum of kernels
        gp_i = lambda i_list: gpmodel_i(x=x, y=y, k_collection=k_collection, indices_kernels=i_list)
        indices = list(range(len(decomp)))  #  + [list(range(len(decomp)))]
        gpmodel = list(map(gp_i, indices))
        kernel_out = kernels
        # for j in range(len(gpmodel)):
        #     gpmodel[0].kern.kernels[j].lengthscales.prior = gpflow.priors.Gamma(shape=1., scale=1.)
        # #     gpmodel[0].kern.kernels[j].variance.prior = gpflow.priors.Gamma(shape=3., scale=2.)
        # for gpmodel_j in gpmodel:
        #     gpmodel_j.compile()
    if model == 'QGPR':
        with gpflow.defer_build():
            kernel_attr = import_attr('gpflow/kernels', attribute=kernel)
            kernel_out = kernel_attr(input_dim=input_dim, ARD=ARD)
            from tfbo.models.gpr_models import QGPR_stable
            gpmodel = QGPR_stable(X=x, Y=y, kern=kernel_out, quantile=quantile)
            gpmodel.kern.lengthscales.prior = gpflow.priors.Gamma(shape=1., scale=1.)
            # gpmodel.kern.variance.prior = gpflow.priors.Gamma(shape=7.5, scale=1.)
            gpmodel.compile()
    return kernel_out, gpmodel



def initialize_m_models(x, y, input_dim, model, kernel, ARD, nn=None, decomp=None, **kwargs):
    '''
    Initializer for Manifold GP Autoencoder. It initializes the manifold GP for encoder and Multi-output GPs for decoder
    '''
    kernel_attr = import_attr('gpflow/kernels', attribute=kernel)
    if model == 'encoder':
        # Return the kernel and the manifold GP used for learning the nonlinear embedding z=NN(X). Non lists!
        # input_dim: proj_dim, the kernel operates in a low-dimensional space
        # nn is required! decomp is None!
        assert nn is not None and decomp is None
        kern_out = kernel_attr(input_dim=input_dim, ARD=ARD, lengthscales=np.ones(shape=[input_dim])*0.2)
        gp_out = MGPR(X=x, Y=y, kern=kern_out, nn=nn)
    elif model == 'decoder':
        # Return a list of kernels and multi-output GPs for learning the mapping to the original space.
        # input_dim: proj_dim, the kernel operates in a low-dimensional space
        # output_dim: number of output features for each MoGP
        # decomp: list of lists, each low-level list corresponds to indices of features mapped by the MoGP
        # decomp is required! nn is None!
        assert decomp is not None and nn is None
        output_dim = len(decomp[0])
        def _kern():
            return kernel_attr(input_dim=input_dim, ARD=ARD, active_dims=list(range(input_dim)), lengthscales=np.ones(shape=[input_dim])*0.2) * \
                   gpflow.kernels.Coregion(input_dim=1, output_dim=output_dim, rank=output_dim, active_dims=[input_dim])
        kern_out = [_kern() for _ in range(len(decomp))]

        X_aug = np.vstack([np.hstack([x.copy(), np.ones(shape=[x.shape[0], 1]) * i]) for i in range(output_dim)])
        def _MoGP(decomp_i, kern_i):
            # kern_i.as_pandas_table()
            np.random.seed(23)
            kern_i.kernels[1].W = np.random.randn(output_dim, output_dim)
            # kern_i.as_pandas_table()
            Y = y[:, decomp_i].copy()
            Y_aug = np.vstack([Y[:, i].copy()[:, None] for i in range(len(decomp_i))])
            MoGP_i = Stable_GPR(X=X_aug, Y=Y_aug, kern=kern_i)
            MoGP_i.likelihood.variance = 1e-06  # 0.001
            return MoGP_i
        gp_out = list(map(_MoGP, decomp, kern_out))
    elif model == 'joint':
        output_dim = len(decomp[0])
        def _kern_new():
            return kernel_attr(input_dim=input_dim, ARD=ARD, active_dims=list(range(input_dim)), lengthscales=np.ones(shape=[input_dim])*0.2) * \
                   gpflow.kernels.Coregion(input_dim=1, output_dim=output_dim, rank=output_dim, active_dims=[input_dim])
        kern_out = [_kern_new() for _ in range(len(decomp))]

        for kern_i in kern_out:
            np.random.seed(23)
            kern_i.kernels[1].W = np.random.randn(output_dim, output_dim)

        kern_last = kernel_attr(input_dim=input_dim, ARD=ARD, lengthscales=np.ones(shape=[input_dim]) * 0.2)
        kern_out.append(kern_last)
        kern_joint = Kstack(kern_out)
        gp_out = NN_MoGPR(X=x, Y=y, kern=kern_joint, nn=nn, Mo_dim=output_dim)
    elif model == 'joint_Full':
        # Assuming kern in input is "Multiple_k" kernel with:
        # kern.K(, i=0) = gpflow.kernels.Coregion           Coregionalization kernel MOGP
        # kern.K(, i=1) = gpflow.kernels.Matern52/RBF/etc.  Standard kernel MOGP
        # kern.K(, i=2) = gpflow.kernels.Matern52/RBF/etc.  Standard kernel Manifold GP
        output_dim = x.shape[1]
        kern_out = []
        # [0] Coregionalization kernel
        kern_out.append(gpflow.kernels.Coregion(input_dim=1, output_dim=output_dim, rank=output_dim, active_dims=[
            input_dim]))  # input_dim = proj_dim, output_dim = x.shape[1]
        np.random.seed(23)
        kern_out[0].W = np.random.randn(output_dim, output_dim)
        # [1] Standard kernel MOGP 
        kern_out.append(kernel_attr(input_dim=input_dim, ARD=ARD, active_dims=list(range(input_dim)),
                                    lengthscales=np.ones(shape=[input_dim]) * 0.2))
        # [2] Standard kernel Manifold GP
        kern_out.append(kernel_attr(input_dim=input_dim, ARD=ARD, lengthscales=np.ones(shape=[input_dim]) * 0.2))

        kern_joint = Multiple_k(kern_out)
        gp_out = NN_FullMoGP(X=x, Y=y, kern=kern_joint, nn=nn, Mo_dim=output_dim)
    elif model == 'BLR':
        output_dim = x.shape[1]
        kern_out = []
        # [0] Standard kernel MOGP
        kern_out.append(kernel_attr(input_dim=input_dim, ARD=ARD, active_dims=list(range(input_dim)),
                                    lengthscales=np.ones(shape=[input_dim]) * 0.5))
        # [1] Standard kernel Manifold GP
        kern_out.append(kernel_attr(input_dim=input_dim, ARD=ARD, lengthscales=np.ones(shape=[input_dim]) * 0.5))

        kern_joint = Multiple_k(kern_out)
        alpha_param = alpha()
        # p = output_dim
        p = int(250)
        np.random.seed(123)
        sample_train = np.random.normal(loc=0., scale=1., size=[x.shape[0], p])
        sample_test = np.random.normal(loc=0., scale=1., size=[1, p])
        gp_out = NN_BLRMoGP(X=x, Y=y, kern=kern_joint, nn=nn, Mo_dim=output_dim, alpha=alpha_param, sample_train=sample_train, sample_test=sample_test, p=p)
    else:
        raise ValueError('Model specified not implemented')

    return kern_out, gp_out

def initialize_acquisition(loss, gpmodel, ymin=None, **kwargs):
    '''
    Initialize acquisition functions as inputs to the optimizer, kwargs specify gpmodel and ymin at each iteration
    Make acquisition a function of x
    :param loss:
    :param gpmodel: update at each iteration
    :param ymin: update at each iteration
    :param kwargs:
    :return:
    '''
    if loss == 'Neg_ei':
        from tfbo.acquisitions.acquisition_functions import Neg_ei_wg
        acquisition = lambda x: Neg_ei_wg(x, gpmodel=gpmodel, ymin=ymin)
    elif loss == 'Neg_pi':
        from tfbo.acquisitions.acquisition_functions import Neg_pi_wg
        acquisition = lambda x: Neg_pi_wg(x, gpmodel=gpmodel, ymin=ymin)
    elif loss == 'lcb':
        from tfbo.acquisitions.acquisition_functions import lcb_wg
        acquisition = lambda x: lcb_wg(x, gpmodel=gpmodel)
    return acquisition


def initialize_acquisition_mo(loss, gpmodel, input_dim, ymin=None, **kwargs):
    '''
    Initialize acquisition functions as inputs to the optimizer, kwargs specify gpmodel and ymin at each iteration
    Make acquisition a function of x
    :param loss:
    :param gpmodel: update at each iteration
    :param ymin: update at each iteration
    :param kwargs:
    :return:
    '''
    if loss == 'Neg_ei':
        from tfbo.acquisitions.acquisition_functions import Neg_ei_mo
        acquisition = lambda x: Neg_ei_mo(x, gpmodel=gpmodel, ymin=ymin, input_dim=input_dim)
    elif loss == 'Neg_pi':
        from tfbo.acquisitions.acquisition_functions import Neg_pi_mo
        acquisition = lambda x: Neg_pi_mo(x, gpmodel=gpmodel, ymin=ymin, input_dim=input_dim)
    elif loss == 'lcb':
        from tfbo.acquisitions.acquisition_functions import lcb_mo
        acquisition = lambda x: lcb_mo(x, gpmodel=gpmodel, input_dim=input_dim)
    return acquisition


def initialize_acquisition_dgp(loss, gpmodel, ymin=None, num_samples=1, **kwargs):
    '''
    Initialize acquisition functions as inputs to the optimizer, kwargs specify gpmodel and ymin at each iteration
    Make acquisition a function of x
    :param loss:
    :param gpmodel: update at each iteration
    :param ymin: update at each iteration
    :param kwargs:
    :return:
    '''
    if loss == 'Neg_ei':
        from tfbo.acquisitions.acquisition_functions import Neg_ei_dgp
        acquisition = lambda x: Neg_ei_dgp(x, gpmodel=gpmodel, ymin=ymin, num_samples=num_samples)
    elif loss == 'Neg_pi':
        from tfbo.acquisitions.acquisition_functions import Neg_pi_dgp
        acquisition = lambda x: Neg_pi_dgp(x, gpmodel=gpmodel, ymin=ymin, num_samples=num_samples)
    elif loss == 'lcb':
        from tfbo.acquisitions.acquisition_functions import lcb_dgp
        acquisition = lambda x: lcb_dgp(x, gpmodel=gpmodel, num_samples=num_samples)
    return acquisition