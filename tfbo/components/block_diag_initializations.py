import numpy as np
import gpflow
from tfbo.utils.import_modules import import_attr
from tfbo.models.cov_funcs import Collection, Kstack, Multiple_k, Kparallel
from tfbo.models.gplvm_models import MGPR, Stable_GPR, FastNN_MoGPR, NN_FullMoGP


def bloc_diag_initialize_models(x, y, input_dim, model, kernel, ARD, nn=None, decomp=None, **kwargs):
    '''
    Initializer for Manifold GP Autoencoder. It initializes the manifold GP for encoder and Multi-output GPs for decoder
    '''
    kernel_attr = import_attr('gpflow/kernels', attribute=kernel)
    if model == 'joint':
        # Joint Manifold GP and Manifold MOGP model. The Manifold MOGP assumes independence between subsets of dimensions (components).
        # In this mode 'joint', for each component a different base kernel is defined.
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
        kern_joint = Kparallel(kern_out)
        gp_out = FastNN_MoGPR(X=x, Y=y, kern=kern_joint, nn=nn, Mo_dim=output_dim)  # Mo_dim = int(3)
    elif model == 'diagonal_joint':
        # Joint Manifold GP and Manifold MOGP model. The Manifold MOGP assumes independence between subsets of dimensions (components).
        # In this model 'diagonal_joint', for each component the same base kernel is defined "single_kernel".
        output_dim = len(decomp[0])
        single_kernel = kernel_attr(input_dim=input_dim, ARD=ARD, active_dims=list(range(input_dim)),
                    lengthscales=np.ones(shape=[input_dim]) * 0.2)
        def _kern_new():
            return single_kernel * gpflow.kernels.Coregion(input_dim=1, output_dim=output_dim, rank=output_dim,
                                                active_dims=[input_dim])
        kern_out = [_kern_new() for _ in range(len(decomp))]


        for kern_i in kern_out:
            np.random.seed(23)
            kern_i.kernels[1].W = np.random.randn(output_dim, output_dim)

        kern_last = kernel_attr(input_dim=input_dim, ARD=ARD, lengthscales=np.ones(shape=[input_dim]) * 0.2)
        kern_out.append(kern_last)
        kern_joint = Kparallel(kern_out)
        gp_out = FastNN_MoGPR(X=x, Y=y, kern=kern_joint, nn=nn, Mo_dim=output_dim)
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
    else:
        raise ValueError('Model specified not implemented')

    return kern_out, gp_out


def block_diag_initialize_acquisition(loss, gpmodel, ymin=None, **kwargs):
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