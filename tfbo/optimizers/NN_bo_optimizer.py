import gpflow
import numpy as np
import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.components.initializations import initialize_m_models
from tfbo.components.block_diag_initializations import block_diag_initialize_acquisition
from tfbo.components.block_diag_initializations import bloc_diag_initialize_models
from tfbo.models.gplvm_models import Ort_NN
from scipy.optimize import minimize
from scipy.stats import norm


class NN_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        # self.initialize_normalization()
        self.initialize_probit()
        self.identity = False
        self.Mo_dim = int(3)
        self.decomposition = [list(np.arange(start=i*self.Mo_dim, stop=(i+1)*self.Mo_dim, step=1)) for i in
                              range(int(np.floor(self.input_dim/self.Mo_dim)))]
        self.Xnn = None
        self.latent_bound = []
        self.latent_grid = []


    def initialize_modelM(self):

        nn = Ort_NN(dims=[self.Xprobit.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
                    name=None)

        k_list, gp_nnjoint = bloc_diag_initialize_models(x=np.copy(self.Xprobit), y=np.copy(self.Ynorm),
                                                 input_dim=self.proj_dim,
                                                 model='joint',
                                                 kernel='Matern52',
                                                 ARD=True,
                                                 nn=nn,
                                                 decomp=self.decomposition)     # last kernel is the Y kernel

        # k2_list, gp_nnjoint_test = initialize_m_models(x=np.copy(self.Xprobit), y=np.copy(self.Ynorm),
        #                                          input_dim=self.proj_dim,
        #                                          model='joint',
        #                                          kernel='Matern52',
        #                                          ARD=True,
        #                                          nn=nn,
        #                                          decomp=self.decomposition)     # last kernel is the Y kernel

        gp_nnjoint.likelihood.variance = 1e-06  # 0.001
        # gp_nnjoint_test.likelihood.variance = 1e-06

        return k_list, gp_nnjoint, nn


    def generate_x(self, x_proj, gp_joint):

        # mean_new = gp_joint.predict_x(x_proj)
        mean_new, post_cov = gp_joint.predict_x(x_proj)
        joint_new = np.zeros(shape=[1, self.Xprobit.shape[1]])
        joint_var = np.zeros(shape=[1, self.Xprobit.shape[1]])
        for decomp_jj, jj in zip(self.decomposition, list(range(len(self.decomposition)))):
            joint_new[:, decomp_jj] = np.copy(mean_new[jj, :, 0])
            joint_var[:, decomp_jj] = np.copy(np.diag(post_cov[jj, :, :]))

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
            print('post_cov')
            print(post_cov)
            print('x_out')
            print(x_out)
            print('alpha_reshape')
        # x_new = (joint_new * self.X_std) + self.X_mean     # originally mapped to Xnorm
        # x_out = np.clip(x_out, a_min=0., a_max=1.)
        return x_out


    def update_latent_bounds(self, opt_config):
        m_min = np.clip(np.min(self.Xnn, axis=0, keepdims=True) - 0.2, a_min=0., a_max=1.)  # refined
        m_max = np.clip(np.max(self.Xnn, axis=0, keepdims=True) + 0.2, a_min=0., a_max=1.)
        self.latent_bound = []
        for i in range(self.proj_dim):
            self.latent_bound += [(m_min[0, i].copy(), m_max[0, i].copy())]
        # # uncomment if want to use L-BFGS-B with bounds on the optimization variable
        opt_config['bounds'] = self.latent_bound * self.num_init

        self.latent_grid = np.multiply(np.copy(self.grid), m_max - m_min) + m_min
        return opt_config


    def run(self, maxiters=20):
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')  # check import configuration
        for j in range(maxiters):
            print('iteration: ', j)

            self.reset_graph()
            # initialize model
            k_list, gp_nnjoint, nn = self.initialize_modelM()
            try:
                gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
            except:
                try:
                    gp_nnjoint.likelihood.variance = 1e-03
                    gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
                except:
                    print('Failure in optimization of hyper-parameters, reset to standard ones')

            Xnn = gp_nnjoint.nn.np_forward(self.Xprobit)
            self.Xnn = Xnn
            self.hyps.append(self.get_hyps(gp_nnjoint))

            # optimize the acquisition function within bounds
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = block_diag_initialize_acquisition(loss=self.loss, gpmodel=gp_nnjoint, **kwargs)
            opt_config = self.update_latent_bounds(opt_config)
            x_proj_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)

            x_tp1 = self.generate_x(x_proj_tp1, gp_nnjoint)

            y_tp1 = self.evaluate(x_tp1)
            self.update_data(x_tp1, y_tp1)
        lik = []

        return self.data_x, self.data_y, self.hyps, lik

    def minimize_acquisition(self, acquisition, opt_config):

        acquisition_grid, acq_sum, acq_grad = acquisition(self.latent_grid)
        indices_sorted = np.argsort(acquisition_grid, axis=0)
        x_topk = np.ravel(np.copy(self.latent_grid[indices_sorted[:self.num_init, 0], :]))

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

    def update_data(self, x_new, y_new):
        self.data_x = np.concatenate([self.data_x, x_new], axis=0)     # check shapes
        self.data_y = np.concatenate([self.data_y, y_new], axis=0)
        # self.initialize_normalization()
        self.initialize_probit()

    def get_hyps(self, gp):
        lengthscales = gp.kern.kernels[-1].lengthscales.read_value()
        kern_var = gp.kern.kernels[-1].variance.read_value()[None]
        noise_var = gp.likelihood.variance.read_value()[None]
        return np.concatenate([lengthscales, kern_var, noise_var], axis=0)