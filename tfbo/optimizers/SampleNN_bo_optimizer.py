import gpflow
import numpy as np
import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.components.initializations import initialize_acquisition
from tfbo.components.initializations import initialize_m_models
from tfbo.models.gplvm_models import NN, Stable_GPR, Ort_NN
from scipy.optimize import minimize


class SampleNN_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.initialize_X()  # check correct shapes
        self.identity = False
        self.Mo_dim = self.data_x.shape[1]
        self.Xnn = None
        self.latent_bound = []
        self.latent_grid = []
        self.L = 1.


    def initialize_modelM(self):


        nn = Ort_NN(dims=[self.data_x.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
                    name=None)

        k_list, gp_nnjoint = initialize_m_models(x=np.copy(self.Xnorm), y=np.copy(self.Ynorm),
                                                 input_dim=self.proj_dim,
                                                 model='joint_Full',
                                                 kernel='Matern52',
                                                 ARD=True,
                                                 nn=nn)     # last kernel is the Manifold GP kernel

        gp_nnjoint.likelihood.variance = 1e-06  # 0.001

        return k_list, gp_nnjoint, nn


    def generate_x(self, x_proj, gp_joint):

        # mean_new, cov_new = gp_joint.predict_x(x_proj)
        posterior_sample = gp_joint.sample_x(x_proj)

        x_new = (posterior_sample * self.X_std) + self.X_mean
        return self.sigmoid(x_new)          # check shape


    def initialize_X(self):

        self.Y_mean = np.mean(self.data_y, axis=0, keepdims=True)
        self.Y_std = np.std(self.data_y, axis=0, keepdims=True)

        self.X_inf = self.logit(self.data_x)
        self.X_mean = np.mean(self.X_inf, axis=0, keepdims=True)
        self.X_std = np.std(self.X_inf, axis=0, keepdims=True)

        self.Xnorm = (self.X_inf - self.X_mean) / self.X_std
        # self.X_mean = np.mean(self.data_x, axis=0, keepdims=True)
        # self.X_std = np.std(self.data_x, axis=0, keepdims=True)
        #
        # self.Xnorm = (self.data_x - self.X_mean) / self.X_std
        self.Ynorm = (self.data_y - self.Y_mean) / self.Y_std   # check all shapes

        if np.isnan(self.Ynorm).any(): raise ValueError('Ynorm contains nan')
        def test_(input_norm):
            xm_test = np.mean(input_norm, axis=0, keepdims=True)
            xst_test = np.std(input_norm, axis=0, keepdims=True) - np.ones([1, input_norm.shape[1]])
            boolm_list = [np.abs(xm_i[0]) <= 1e-02 for xm_i in list(xm_test.transpose())]
            bools_list = [np.abs(xs_i[0]) <= 1e-02 for xs_i in list(xst_test.transpose())]
            assert all(boolm_list) and all(bools_list)
        test_(self.Xnorm)
        test_(self.Ynorm)


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
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')
        for j in range(maxiters):
            print('iteration: ', j)

            self.reset_graph()

            # initialize model
            k_list, gp_nnjoint, nn = self.initialize_modelM()
            gp_nnjoint.kern.kernels[1].variance = 10.

            try:
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

            # optimize the acquisition function within 0,1 bounds
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition(loss=self.loss, gpmodel=gp_nnjoint, **kwargs)

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
        self.initialize_X()

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