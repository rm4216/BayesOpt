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


class SRNNKL_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, seed, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.initialize_normalization()  # check correct shapes
        self.identity = False
        self.Mo_dim = self.Xnorm.shape[1]
        self.Xnn = None
        self.latent_bound = []
        self.latent_grid = []
        self.L = 1.
        self.seed = seed


    def initialize_modelM(self):


        # nn = Ort_NN(dims=[self.Xnorm.shape[1], 5, 10, 5, self.proj_dim], N=0, proj_dim=0,
        #             name=None)
        nn = NN(dims=[self.Xnorm.shape[1],  5, 10, 5, self.proj_dim], N=0, proj_dim=0,
                    name=None)

        k_list, gp_nnjoint = initialize_m_models(x=np.copy(self.Xnorm), y=np.copy(self.Ynorm),
                                                 input_dim=self.proj_dim,
                                                 model='joint_Full',
                                                 kernel='Matern52',
                                                 ARD=True,
                                                 nn=nn)     # last kernel is the Manifold GP kernel

        gp_nnjoint.likelihood.variance = 0.1 # 1e-06  # 0.001

        return k_list, gp_nnjoint, nn


    def generate_x(self, x_proj, gp_joint):

        # num_outputs = len(self.decomposition[0])
        # fmean, fvar = gp_joint.predict_x(self.Xnn)
        # fmean_reshape = np.reshape(fmean, [len(self.decomposition), num_outputs, self.Xnn.shape[0]]).transpose((0, 2, 1))
        # x_mean = np.zeros(shape=[self.Xnn.shape[0], self.Xnorm.shape[1]])
        # for decomp_kk, kk in zip(self.decomposition, list(range(len(self.decomposition)))):
        #     x_mean[:, decomp_kk] = fmean_reshape[kk, :, :]
        #
        # errs_x_mean = np.mean(np.abs(x_mean - self.Xnorm), axis=0)

        # mean_new, var_new = gp_joint.predict_x(x_proj)
        mean_new = gp_joint.predict_x(x_proj)
        joint_new = np.transpose(mean_new)
        # joint_new = np.zeros(shape=[1, self.Xnorm.shape[1]])
        # for decomp_jj, jj in zip(self.decomposition, list(range(len(self.decomposition)))):
        #     joint_new[:, decomp_jj] = mean_new[jj, :, 0]

        x_new = (joint_new * self.X_std) + self.X_mean     # originally mapped to Xnorm
        x_out = np.clip(x_new, a_min=0., a_max=1.)
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

            x_mean_fast = np.copy(self.Xnorm[ind_minimizers, :])

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
            # Xnn0 = gp_nnjoint.nn.np_forward(self.Xnorm)
            # kc = k_list[1] * k_list[0]
            # Xnn_test = np.concatenate(
            #     [np.concatenate([Xnn0, np.ones(shape=[np.shape(Xnn0)[0], 1]) * i], axis=1)
            #      for i in range(gp_nnjoint.Mo_dim)], axis=0)
            # Y_test = np.concatenate([self.Xnorm[:, i][:, None] for i in range(gp_nnjoint.Mo_dim)], axis=0)
            # gp_test = gpflow.models.GPR(X=Xnn_test, Y=Y_test, kern=kc)
            # gp_test.likelihood.variance = 1e-06
            # lik_test = gp_test.compute_log_likelihood()
            # gpm = gpflow.models.GPR(X=Xnn0, Y=self.Ynorm, kern=k_list[2])
            # gpm.likelihood.variance = 1e-06
            # likm = gpm.compute_log_likelihood()
            # lik_err = np.abs(lik_nn-lik_test-likm)

            gp_nnjoint.nn.W_0 = np.array([[-0.72257252, 0.26017714],
                                          [-1.23547449, -0.87398094],
                                          [-0.05795134, 0.22184529],
                                          [-4.33704576, -1.03866942],
                                          [4.16884434, 0.1687948]])
            gp_nnjoint.nn.W_1 = np.array([[-0.17611191, 0.84349685, 1.44230698, 0.18555664, -0.19708862],
                                          [-0.13689745, 1.86417045, 2.33110755, 1.20521291, 0.71162644],
                                          [0.47687133, 0.31373425, -1.1891341, 2.18089067, -3.93909819],
                                          [-0.2272015, 1.93327611, -1.57774183, -1.26255085, -0.15080552],
                                          [-0.4890983, -1.81724449, -1.65700209, -0.75827901, 1.64434325],
                                          [0.10663821, -0.12244555, 2.26286785, -0.88992352, 2.63438025],
                                          [-1.14518348, -2.48144707, -0.35203317, 0.23830179, 0.0816695],
                                          [-0.5185169, 2.43075116, 0.09996988, 1.56821543, 2.57299817],
                                          [1.27373299, -2.17523897, 2.56801105, -1.29495389, -1.38732749],
                                          [2.16933267, -0.82218552, 1.94225155, 3.44593108, 1.76706837]])
            gp_nnjoint.nn.W_2 = np.array([[-1.06815199, 0.67328749, 1.33295767, -0.82976342, 1.08580199,
                                           0.07772985, -0.45765023, -0.05497667, -2.4756558, 0.08808674],
                                          [0.85855821, -0.10785176, 1.40417131, -1.4510554, -2.43215512,
                                           0.58832488, -0.31426693, 0.88093524, -0.18911669, -1.21866324],
                                          [0.8989253, -0.04077404, 4.74024619, -0.25097489, -0.68791512,
                                           -2.8158515, -1.05096808, -1.15249423, 2.40093649, 2.84014738],
                                          [1.71409331, 0.21485905, 0.47611273, 3.44473025, -0.1917658,
                                           3.08725273, -0.97657774, 0.22685569, 0.33642754, 0.69626424],
                                          [0.60789342, -2.02719287, 0.43644935, 2.13129863, -0.4946168,
                                           0.3486837, -0.02468686, -2.11012978, 0.80318346, -2.0538133]])
            gp_nnjoint.nn.W_3 = np.array([[-1.17012522, 1.4669893, -2.33431889, 4.54361068, 0.219858]])
            gp_nnjoint.nn.b_0 = np.array([[-1.95648467],
                                          [-0.40078642],
                                          [0.03963978],
                                          [-3.13848025],
                                          [0.89017789]])
            gp_nnjoint.nn.b_1 = np.array([[0.84520059],
                                          [0.5069299],
                                          [-1.45844994],
                                          [0.32032038],
                                          [0.94691029],
                                          [0.87558343],
                                          [-0.41215514],
                                          [0.13526481],
                                          [-1.00605875],
                                          [-0.02132958]])
            gp_nnjoint.nn.b_2 = np.array([[-0.11726942],
                                          [0.14056033],
                                          [1.38538488],
                                          [1.71165805],
                                          [-0.41426653]])
            gp_nnjoint.nn.b_3 = np.array([[1.19480249]])
            # gp_nnjoint.nn.trainable = False


            # gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
            try:
                gp_nnjoint.kern.kernels[1].lengthscales = np.ones(shape=[self.proj_dim])
                gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
            except:
                try:
                    gp_nnjoint.kern.kernels[1].lengthscales = np.ones(shape=[self.proj_dim])
                    gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
                except:
                    print('Failure in optimization of hyper-parameters, reset to standard ones')

            Xnn = gp_nnjoint.nn.np_forward(self.Xnorm)
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
        self.initialize_normalization()

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