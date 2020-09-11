import numpy as np
import gpflow
from tfbo.components.initializations import initialize_acquisition, initialize_models
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.utils.import_modules import import_attr


class qgp_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, quantile=0.1, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.decomposition = [np.arange(start=i * self.proj_dim, stop=(i+1) * self.proj_dim) for i in
                              range(int(np.floor(self.input_dim/self.proj_dim)))]
        self.initialize_normalization()
        self.hyps_temp = []
        self.hyps_opt = []
        self.log_lik_opt = []
        self.quantile = quantile

    def initialize_qgp(self, Xi):
        kwargs = {'quantile': self.quantile}
        return initialize_models(x=np.copy(Xi), y=np.copy(self.Ynorm), input_dim=self.proj_dim,
                                 model='QGPR', kernel='Matern52', ARD=True, **kwargs)

    def select_component(self, i):
        return np.copy(self.Xnorm[:, self.decomposition[i]])

    # def assign_hyps(self, gp, hyps_array):
    #     gp.kern.lengthscales = hyps_array[:-2]
    #     gp.kern.variance = hyps_array[-2]
    #     gp.likelihood.variance = hyps_array[-1]
    #     return gp

    def fit_gp(self, gp):
        self.hyps_opt = []
        signal_variance = np.random.uniform(low=0., high=10, size=10)[:, None]
        noise_variance = signal_variance * 0.01
        log_lik = []
        for signal_i, noise_i in zip(list(signal_variance), list(noise_variance)):
            gp = self.reset_hyps(gp)
            gp.kern.variance = signal_i[0]
            gp.likelihood.variance = noise_i[0]
            try:
                gpflow.train.ScipyOptimizer().minimize(gp)
            except:
                gp = self.reset_hyps(gp)
            log_lik.append(gp.compute_log_likelihood()[0][0])
            self.hyps_opt.append(self.get_hyps(gp))

        np_log_liks = np.array(log_lik)
        index_opt = np.argmax(np_log_liks)
        gp = self.assign_hyps(gp, self.hyps_opt[index_opt])
        self.log_lik_opt.append(np_log_liks[index_opt])
        return gp

    def fit_gp_fast(self, gp):
        self.hyps_opt = []
        signal_variance = np.random.uniform(low=0., high=10, size=1)[:, None]
        noise_variance = signal_variance * 0.01
        log_lik = []
        for signal_i, noise_i in zip(list(signal_variance), list(noise_variance)):
            gp = self.reset_hyps(gp)
            gp.kern.variance = signal_i[0]
            gp.likelihood.variance = noise_i[0]
            try:
                gpflow.train.ScipyOptimizer().minimize(gp)
            except:
                gp = self.reset_hyps(gp)
            log_lik.append(gp.compute_log_likelihood()[0][0])
            self.hyps_opt.append(self.get_hyps(gp))

        np_log_liks = np.array(log_lik)
        index_opt = np.argmax(np_log_liks)
        gp = self.assign_hyps(gp, self.hyps_opt[index_opt])
        self.log_lik_opt.append(np_log_liks[index_opt])
        return gp

    def optimize_i(self, gp_i, i, opt_config):
        # gp_i = self.reset_hyps(gp_i)
        # gp_i = self.fit_gp(gp_i)
        gp_i = self.fit_gp_fast(gp_i)
        # try:
        #     # gp_i.kern.lengthscales = np.array([1.,2.])
        #     # gp_i.kern.variance = 3.
        #     # gp_i.likelihood.variance = 4.
        #     # o = gpflow.train.ScipyOptimizer()
        #     # scipyopt = o.make_optimize_tensor(gp_i)
        #     # scipyopt.minimize(maxiter=100)
        #     # gp_i.kern.variance = 5.
        #     # gp_i.likelihood.variance = 0.0001
        #     # gp_i = self.fit_gp(gp_i)
        #     gpflow.train.ScipyOptimizer().minimize(gp_i)
        # except:
        #     gp_i = self.reset_hyps(gp_i)
        self.hyps_temp.append(self.get_hyps(gp_i))

        kwargs = {'ymin': self.Ynorm.min()}
        acquisition = initialize_acquisition(loss=self.loss, gpmodel=gp_i, **kwargs)
        acquisition_normalized = lambda x: self.acquisition_norm(acquisition, x,
                                                                 np.copy(self.X_mean[:, self.decomposition[i]]),
                                                                 np.copy(self.X_std[:, self.decomposition[i]]))
        x_opt, acq_opt = self.minimize_acquisition(acquisition_normalized, opt_config)
        return x_opt, acq_opt

    def run(self, maxiters):
        list_i = list(range(len(self.decomposition)))
        # decompose normalized dataset
        list_xnorm = list(map(self.select_component, list_i))
        list_km = list(map(self.initialize_qgp, list_xnorm))
        list_kernel = [km_i[0] for km_i in list_km]
        list_models = [km_i[1] for km_i in list_km]

        # load optimizer options
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')
        opt_config['bounds'] = [(0., 1.)] * self.proj_dim * self.num_init

        optimize_acq_i = lambda gp_i, i: self.optimize_i(gp_i, i, opt_config)

        for j in range(maxiters):
            print(j)

            list_xa = list(map(optimize_acq_i, list_models, list_i))

            list_x = [xa_i[0] for xa_i in list_xa]
            self.hyps.append(self.collect_hyps())
            x_out = self.compose_x(list_x)
            y_out = self.evaluate(list_x)
            self.update_data(x_out, y_out)
            list_xnorm = list(map(self.select_component, list_i))
            self.reset_graph()
            list_km = list(map(self.initialize_qgp, list_xnorm))
            list_models = [km_i[1] for km_i in list_km]
            # list_models = self.update_models(list_models, list_xnorm)
        return self.data_x, self.data_y, self.hyps, self.log_lik_opt

    def update_data(self, x_out, y_out):
        # update overall dataset and normalize the updated data
        self.data_x = np.concatenate([self.data_x, x_out], axis=0)
        self.data_y = np.concatenate([self.data_y, y_out], axis=0)
        self.initialize_normalization()

    def update_models(self, gpmodels, xnorms):
        for gp_i, x_i in zip(gpmodels, xnorms):
            gp_i.X = np.copy(x_i)
            gp_i.Y = np.copy(self.Ynorm)
        return gpmodels

    def compose_x(self, x_list):
        x0 = np.zeros(shape=[x_list[0].shape[0], self.proj_dim * len(x_list)])  # check shapes of x_i
        for x_i, indices_i in zip(x_list, self.decomposition):
            x0[:, indices_i] = x_i  # if I change x_i changes x0?
        return x0

    def evaluate(self, x_list):
        x_tp1 = self.compose_x(x_list)  # check shape
        if self.input_dim == self.proj_dim:
            y_tp1 = self.objective.f(x_tp1, noisy=True, fulldim=True)
        else:
            y_tp1 = self.objective.f(x_tp1, noisy=True, fulldim=False)
        return y_tp1

    def collect_hyps(self):
        hyps_lengthscales = [hyp_i[:self.proj_dim] for hyp_i in self.hyps_temp]
        hyps_ker_var = [hyp_i[-2] for hyp_i in self.hyps_temp]
        hyps_lik_var = [hyp_i[-1] for hyp_i in self.hyps_temp]

        self.hyps_temp = []
        return np.concatenate([np.ravel(np.array(hyps_lengthscales)), np.array(hyps_ker_var), np.array(hyps_lik_var)],
                              axis=0)