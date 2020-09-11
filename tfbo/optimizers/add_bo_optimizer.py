from tfbo.optimizers.optimizer_class import optimizer
import numpy as np
from tfbo.components.initializations import initialize_models, initialize_acquisition
from tfbo.utils.import_modules import import_attr
import gpflow
from scipy.optimize import minimize
from collections import OrderedDict
import tensorflow as tf


class add_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.decomposition = [np.arange(start=i * self.proj_dim, stop=(i+1) * self.proj_dim) for i in
                              range(int(np.floor(self.input_dim/self.proj_dim)))]
        self.grid = np.tile(self.grid, reps=[1, len(self.decomposition)])
        self.initialize_normalization()
        self.hyps_opt = []
        self.log_lik_opt = []

    def evaluate(self, x_list):
        x_tp1 = self.compose_x(x_list)  # check shape
        y_tp1 = self.objective.f(x_tp1, noisy=True, fulldim=False)
        return y_tp1

    def compose_x(self, x_list):
        x0 = np.zeros(shape=[x_list[0].shape[0], self.proj_dim * len(x_list)])  # check shapes of x_i
        for x_i, indices_i in zip(x_list, self.decomposition):
            x0[:, indices_i] = x_i  # if I change x_i changes x0?
        return x0

    def update_data(self, x_tp1, y_tp1):
        self.data_x = np.concatenate([self.data_x, x_tp1], axis=0)  # check shapes
        self.data_y = np.concatenate([self.data_y, y_tp1], axis=0)
        self.initialize_normalization()

    def update_model(self, gp):
        for gp_i in gp:
            gp_i.X = self.Xnorm    # check initial data set
            gp_i.Y = self.Ynorm
        return gp

    def reset_hyps(self, gp_i):
        for j in range(len(self.decomposition)):
            gp_i.kern.kernels[j].lengthscales = np.ones(shape=[self.proj_dim])  # check shape, check transformation hyps
            gp_i.kern.kernels[j].variance = 1.
        gp_i.likelihood.variance = 1.
        # gp.read_trainables()
        return gp_i

    # def reset_graph(self):
    #     tf.reset_default_graph()
    #     graph = tf.get_default_graph()
    #     gpflow.reset_default_session(graph=graph)

    def assign_hyps(self, gp, hyps_array):
        for j in range(len(self.decomposition)):
            gp.kern.kernels[j].lengthscales = np.copy(hyps_array[j*self.proj_dim:(j+1)*self.proj_dim])
            gp.kern.kernels[j].variance = hyps_array[self.input_dim+j]
        gp.likelihood.variance = hyps_array[-1]
        return gp

    # def fit_gp(self, gp):
    #     self.hyps_opt = []
    #     signal_variance = np.random.uniform(low=0., high=10, size=10)[:, None]
    #     noise_variance = signal_variance * 0.01
    #     log_lik = []
    #     for signal_i, noise_i in zip(list(signal_variance), list(noise_variance)):
    #         gp = self.reset_hyps(gp)
    #         for j in range(len(self.decomposition)):
    #             gp.kern.kernels[j].variance = signal_i[0]
    #         gp.likelihood.variance = noise_i[0]
    #         try:
    #             gpflow.train.ScipyOptimizer().minimize(gp)
    #         except:
    #             gp = self.reset_hyps(gp)
    #         log_lik.append(gp.compute_log_likelihood())
    #         self.hyps_opt.append(self.get_hyps(gp))
    #
    #     np_log_liks = np.array(log_lik)
    #     index_opt = np.argmax(np_log_liks)
    #     gp = self.assign_hyps(gp, self.hyps_opt[index_opt])
    #     self.log_lik_opt.append(np_log_liks[index_opt])
    #     return gp

    def run(self, maxiters):
        # initialize model
        k_list, gpmodels = initialize_models(np.copy(self.Xnorm), np.copy(self.Ynorm), input_dim=self.proj_dim,
                                             model='AddGPR', kernel='Matern52', ARD=True, decomp=self.decomposition)
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')
        opt_config['bounds'] = [(0., 1.)] * self.input_dim * self.num_init

        for i in range(maxiters):
            print(i)
            gp0 = gpmodels[0]
            try:
                gpflow.train.ScipyOptimizer().minimize(
                    gp0)  # test it trains all GPR models simultaneously: only kern variables not likelihood
            except:
                gp0 = self.reset_hyps(gp0)
            self.hyps.append(self.get_hyps(gp0))

            def opt_i(self, gp_i, gpsum):
                gp_i.likelihood.variance = gpsum.likelihood.variance

                kwargs = {'ymin': self.Ynorm.min()}
                acquisition = initialize_acquisition(loss=self.loss, gpmodel=gp_i, **kwargs)    # alpha(x)
                acquisition_normalized = lambda x: \
                    self.acquisition_norm(acquisition, x, np.copy(self.X_mean), np.copy(self.X_std))     # alpha(xnorm)
                # x_opt, acq_opt = self.minimize_acquisition(acquisition_normalized, opt_config)
                try:
                    x_opt, acq_opt = self.minimize_acquisition(acquisition_normalized, opt_config)
                except:
                    gp_i = self.reset_hyps(gp_i)
                    x_opt, acq_opt = self.minimize_acquisition(acquisition_normalized, opt_config)
                return np.copy(x_opt[:, self.decomposition[gp_i.i[0]]]), acq_opt   # x^(i)_opt, alpha(x^(i)_opt)
            optimize_i = lambda gp_ith: opt_i(self, gp_ith, gpsum=gp0)
            xa_list = list(map(optimize_i, gpmodels))

            x_list = [xi_ai[0] for xi_ai in xa_list]
            x_tp1 = self.compose_x(x_list)
            y_tp1 = self.evaluate(x_list)
            self.update_data(x_tp1, y_tp1)
            # self.update_model(gpmodels)
            self.reset_graph()
            k_list, gpmodels = initialize_models(np.copy(self.Xnorm), np.copy(self.Ynorm), input_dim=self.proj_dim,
                                                 model='AddGPR', kernel='Matern52', ARD=True, decomp=self.decomposition)
        return self.data_x, self.data_y, self.hyps, self.log_lik_opt

    def get_hyps(self, gp):
        def kern_l(index):
            return gp.kern.kernels[index].lengthscales.read_value()
        def kern_v(index):
            return gp.kern.kernels[index].variance.read_value()[None]
        indices = list(range(len(self.decomposition)))
        arr_l = np.array(list(map(kern_l, indices)))
        arr_v = np.array(list(map(kern_v, indices)))
        lik = gp.likelihood.variance.read_value()[None]
        hyps = np.concatenate([np.ravel(arr_l), np.ravel(arr_v), lik], axis=0)
        return hyps

    def minimize_acquisition(self, acquisition, opt_config):
        acquisition_grid, acq_sum, acq_grad = acquisition(self.grid)
        indices_sorted = np.argsort(acquisition_grid, axis=0)
        x_topk = np.ravel(np.copy(self.grid[indices_sorted[:self.num_init, 0], :]))     # check copy necessary

        def acq_objective(self, xopt, acquisition):
            xopt_reshape = np.reshape(xopt, [self.num_init, self.input_dim])
            acq_arr, acq_sum, acq_grad = acquisition(xopt_reshape)
            grad_reshape = np.ravel(acq_grad[0])   # check shape
            return  acq_sum[0, 0], grad_reshape
        sum_acquisition_opt = lambda x: acq_objective(self=self, xopt=x, acquisition=acquisition)
        optimize_result = minimize(sum_acquisition_opt, x_topk, **opt_config)

        x_opt_all = np.reshape(optimize_result.x, newshape=[self.num_init, self.input_dim])
        f_opt_all, f_sum, f_grad = acquisition(x_opt_all)
        x_opt = x_opt_all[f_opt_all.argmin(), :][None]
        f_opt = f_opt_all.min()[None, None]
        return x_opt, f_opt

    # def transfer_hyps2(self, gp_receive, gp_give):
    #     for i in range(10):
    #         gp_receive.kern.kernels[i].lengthscales = gp_give.kern.kernels[i].lengthscales.read_value()
    #         gp_receive.kern.kernels[i].variance = gp_give.kern.kernels[i].variance.read_value()
    #     gp_receive.likelihood.variance = gp_give.likelihood.variance.read_value()
    #     return gp_receive