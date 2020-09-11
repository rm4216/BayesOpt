import gpflow
import numpy as np
import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from collections import OrderedDict
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.components.initializations import initialize_acquisition
from tfbo.components.initializations import initialize_m_models
from tfbo.models.gplvm_models import NN, Stable_GPR, Ort_NN
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import argparse
import scipy.io as sio
# import random
# random.seed(a=123)
from scipy.stats import norm


class FullNNKL_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        # self.initialize_normalization()  # check correct shapes
        self.initialize_probit()
        self.identity = False
        self.Mo_dim = self.Xprobit.shape[1]
        self.Xnn = None
        self.latent_bound = []
        self.latent_grid = []
        self.L = 1.


    def initialize_modelM(self):


        nn = Ort_NN(dims=[self.Xprobit.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
                    name=None)
        # nn = NN(dims=[self.Xprobit.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
        #             name=None)

        k_list, gp_nnjoint = initialize_m_models(x=np.copy(self.Xprobit), y=np.copy(self.Ynorm),
                                                 input_dim=self.proj_dim,
                                                 model='joint_Full',
                                                 kernel='Matern52',
                                                 ARD=True,
                                                 nn=nn)     # last kernel is the Manifold GP kernel

        gp_nnjoint.likelihood.variance = 1e-06  # 0.001

        return k_list, gp_nnjoint, nn


    def generate_x(self, x_proj, gp_joint):

        # num_outputs = len(self.decomposition[0])
        # fmean, fvar = gp_joint.predict_x(self.Xnn)
        # fmean_reshape = np.reshape(fmean, [len(self.decomposition), num_outputs, self.Xnn.shape[0]]).transpose((0, 2, 1))
        # x_mean = np.zeros(shape=[self.Xnn.shape[0], self.Xprobit.shape[1]])
        # for decomp_kk, kk in zip(self.decomposition, list(range(len(self.decomposition)))):
        #     x_mean[:, decomp_kk] = fmean_reshape[kk, :, :]
        #
        # errs_x_mean = np.mean(np.abs(x_mean - self.Xprobit), axis=0)

        # mean_new, var_new = gp_joint.predict_x(x_proj)
        mean_new, cov_new = gp_joint.predict_x(x_proj)
        joint_new = np.transpose(mean_new)
        # joint_new = np.zeros(shape=[1, self.Xprobit.shape[1]])
        # for decomp_jj, jj in zip(self.decomposition, list(range(len(self.decomposition)))):
        #     joint_new[:, decomp_jj] = mean_new[jj, :, 0]
        joint_var = np.diag(cov_new)[None]
        joint_var = np.clip(np.abs(joint_var), a_min=1e-06, a_max=1e09)
        m = 0.
        v = 1.
        z = (joint_new - m) / (v * np.sqrt(1. + joint_var / (v ** 2.)))                    # Equation (3.82) GPML book = Equation (3.25) GPML book
        x_out = norm.cdf(z)

        if np.isnan(x_out.max()):
            print('joint_new')
            print(joint_new)
            print('joint_var')
            print(joint_var)
            print('z')
            print(z)
            print('mean_new')
            print(mean_new)
            print('cov_new')
            print(cov_new)
            print('x_out')
            print(x_out)
            print('alpha_reshape')
        # x_new = (joint_new * self.X_std) + self.X_mean     # originally mapped to Xnorm
        # x_out = np.clip(x_out, a_min=0., a_max=1.)
        return x_out


    def learnLipschitz(self, gp_joint):
        rand_ind = np.random.choice(self.Xnn.shape[0], 10, replace=False)
        z_jac_opt = np.copy(self.Xnn[rand_ind, :])
        jac_Dxd_list = []
        for i in range(z_jac_opt.shape[0]):
            jac_list = []
            for j in range(self.input_dim):
                mu_x, jac_ij = gp_joint.jacobian_Fullx(z_jac_opt[i, :][None], j)
                jac_list.append(jac_ij)
            jac_Dxd_i = np.concatenate(jac_list, axis=0)
            jac_Dxd_list.append(jac_Dxd_i)
        jac_NxDxd = np.stack(jac_Dxd_list, axis=0)
        self.L = np.max(np.abs(jac_NxDxd))
        # norms_of_Dxd = np.stack([np.linalg.norm(j_i, ord=None, axis=None, keepdims=False) for j_i in jac_Dxd_list], axis=0)
        # self.L = np.max(norms_of_Dxd)


    def KLdist(self, xC, X2):
        xCs = np.sum(np.square(xC), axis=1, keepdims=True)          # M x 1
        Xs = np.sum(np.square(X2), axis=1, keepdims=True)           # N x 1
        xXT = np.matmul(xC, np.transpose(X2))                       # M x N
        KL = (xCs + np.transpose(Xs) - 2. * xXT) / 2.               # M x N
        return KL


    def NLconstraint(self, opt_config):

        def KLconstraint(self, x):
            xC = np.reshape(x, [self.num_init, self.proj_dim])          # M x d
            KL = self.KLdist(xC, self.Xnn)                              # M x N
            ind_minimizers = np.argmin(KL, axis=1)                      # (M,)

            x_mean_fast = np.copy(self.Xprobit[ind_minimizers, :])

            max_mean_components = np.amax(np.abs(x_mean_fast), axis=1)[:, None]  # M x 1
            rterm = max_mean_components ** 2. / (2 * self.L**2)                  # M x 1
            lterm = np.min(KL, axis=1)[:, None]                             # M x 1
            constraint = lterm - rterm
            return constraint[:, 0]

        f_constraint = lambda x: KLconstraint(self=self, x=x)
        opt_config['constraints'] = NonlinearConstraint(f_constraint, lb=-np.inf, ub=0, jac='2-point')
        return opt_config


    def run(self, maxiters=20):
        opt_config = import_attr('tfbo/configurations', attribute='KLacquisition_opt')  # check import configuration
        for j in range(maxiters):
            print('iteration: ', j)

            self.reset_graph()

            # initialize model
            k_list, gp_nnjoint, nn = self.initialize_modelM()
            gp_nnjoint.kern.kernels[1].variance = 10.

            # # test likelihood
            # lik_nn = gp_nnjoint.compute_log_likelihood()
            # Xnn0 = gp_nnjoint.nn.np_forward(self.Xprobit)
            # kc = k_list[1] * k_list[0]
            # Xnn_test = np.concatenate(
            #     [np.concatenate([Xnn0, np.ones(shape=[np.shape(Xnn0)[0], 1]) * i], axis=1)
            #      for i in range(gp_nnjoint.Mo_dim)], axis=0)
            # Y_test = np.concatenate([self.Xprobit[:, i][:, None] for i in range(gp_nnjoint.Mo_dim)], axis=0)
            # gp_test = gpflow.models.GPR(X=Xnn_test, Y=Y_test, kern=kc)
            # gp_test.likelihood.variance = 1e-06
            # lik_test = gp_test.compute_log_likelihood()
            # gpm = gpflow.models.GPR(X=Xnn0, Y=self.Ynorm, kern=k_list[2])
            # gpm.likelihood.variance = 1e-06
            # likm = gpm.compute_log_likelihood()
            # lik_err = np.abs(lik_nn-lik_test-likm)


            # gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
            try:
                gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
            except:
                try:
                    gp_nnjoint.likelihood.variance = 1e-03
                    gp_nnjoint.kern.kernels[1].lengthscales = np.ones(shape=[self.proj_dim])
                    gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
                except:
                    print('Failure in optimization of hyper-parameters, reset to standard ones')

            Xnn = gp_nnjoint.nn.np_forward(self.Xprobit)
            self.Xnn = Xnn
            self.hyps.append(self.get_hyps(gp_nnjoint))

            # # test Manifold GP predictions
            # fmean, fvar = gp_nnjoint.predict_f(Xnn)



            # optimize the acquisition function within 0,1 bounds
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition(loss=self.loss, gpmodel=gp_nnjoint, **kwargs)
            # opt_config = self.update_latent_bounds(opt_config)
            if (j%10 == 0):
                self.learnLipschitz(gp_nnjoint)     # learn new Lipschitz constant every 10 iters
            opt_config = self.NLconstraint(opt_config)
            x_proj_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)

            x_tp1 = self.generate_x(x_proj_tp1, gp_nnjoint)

            y_tp1 = self.evaluate(x_tp1)
            self.update_data(x_tp1, y_tp1)
        lik = []

        return self.data_x, self.data_y, self.hyps, lik

    def minimize_acquisition(self, acquisition, opt_config):

        # acquisition_grid, acq_sum, acq_grad = acquisition(self.grid)
        # indices_sorted = np.argsort(acquisition_grid, axis=0)
        KL = self.KLdist(self.grid, self.Xnn)                   # M x N
        KLmin = np.amin(KL, axis=1, keepdims=True)
        indices_sorted = np.argsort(KLmin, axis=0)
        x_topk = np.ravel(np.copy(self.grid[indices_sorted[:self.num_init, 0], :]))     # check copy necessary

        # acquisition_grid, acq_sum, acq_grad = acquisition(self.grid)
        # indices_sorted = np.argsort(acquisition_grid, axis=0)
        # x_topk = np.ravel(np.copy(self.grid[indices_sorted[:self.num_init, 0], :]))     # check copy necessary

        def acq_objective(self, xopt, acquisition):
            xopt_reshape = np.reshape(xopt, [self.num_init, self.proj_dim])
            acq_arr, acq_sum, acq_grad = acquisition(xopt_reshape)
            grad_reshape = np.ravel(acq_grad[0])   # check shape
            return  acq_sum[0, 0], grad_reshape
        sum_acquisition_opt = lambda x: acq_objective(self=self, xopt=x, acquisition=acquisition)
        optimize_result = minimize(sum_acquisition_opt, x_topk, **opt_config)

        x_opt_all = np.reshape(optimize_result.x, newshape=[self.num_init, self.proj_dim])
        f_opt_all, f_sum, f_grad = acquisition(x_opt_all)
        x_opt = x_opt_all[f_opt_all.argmin(), :][None]
        f_opt = f_opt_all.min()[None, None]
        return x_opt, f_opt

    def evaluate(self, x_tp1):
        return self.objective.f(x_tp1, noisy=True, fulldim=self.identity)
        # fx = self.objective.f(x_tp1, noisy=False, fulldim=self.identity)
        # noise = np.array([random.gauss(0, self.objective.scale)])[:, None]   # .normal(loc=0., scale=self.objective.scale, size=fx.shape[0])[:, None]
        # return fx + noise

    def update_data(self, x_new, y_new):
        self.data_x = np.concatenate([self.data_x, x_new], axis=0)     # check shapes
        self.data_y = np.concatenate([self.data_y, y_new], axis=0)
        # self.initialize_normalization()
        self.initialize_probit()

    def get_hyps_mo(self, gp):
        lengthscales = gp.kern.kernels[0].lengthscales.read_value()
        kern_var = gp.kern.kernels[0].variance.read_value()[None]
        noise_var = gp.likelihood.variance.read_value()[None]
        return np.concatenate([lengthscales, kern_var, noise_var], axis=0)

    def reset_hyps_mo(self, gp):
        gp.kern.kernels[0].lengthscales = np.ones(shape=[self.proj_dim])  # check shape, check transformation hyps
        gp.kern.kernels[0].variance = 1.
        gp.likelihood.variance = 1.
        # gp.read_trainables()
        return gp

    def reset_hyps(self, gp):
        gp.kern.lengthscales = np.ones(shape=[self.proj_dim])  # check shape, check transformation hyps
        gp.kern.variance = 1.
        gp.likelihood.variance = 1.
        # gp.read_trainables()
        return gp

    def get_hyps(self, gp):
        lengthscales = gp.kern.kernels[-1].lengthscales.read_value()
        kern_var = gp.kern.kernels[-1].variance.read_value()[None]
        noise_var = gp.likelihood.variance.read_value()[None]
        return np.concatenate([lengthscales, kern_var, noise_var], axis=0)