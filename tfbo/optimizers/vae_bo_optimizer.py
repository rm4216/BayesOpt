import numpy as np
import gpflow
from tfbo.utils.import_modules import import_attr
from tfbo.components.initializations import initialize_models, initialize_acquisition
from scipy.optimize import minimize
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.utils.load_save import load_keras_model
from tfbo.utils.name_file import name_model_vae
from scipy.stats import norm


class vae_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, dict_args, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)

        self.dict_args = dict_args
        # self.initialize_normalization()
        self.initialize_probit()
        self.latent_grid = []
        self.log_lik_opt = []
        self.hyps_opt = []

        self.encoder = []
        self.decoder = []
        self.Z_vae = []

    def update_data(self, x, y):
        self.data_x = np.concatenate([self.data_x, x], axis=0)     # check shapes
        self.data_y = np.concatenate([self.data_y, y], axis=0)
        # self.initialize_normalization()
        self.initialize_probit()

    def evaluate(self, x):
        assert x.shape[1] == self.input_dim
        y = self.objective.f(x, noisy=True, fulldim=False)
        return y

    def load_vae_models(self, path):
        filename = name_model_vae(self.dict_args)
        encoder = load_keras_model(path + filename + '_encoder.h5')
        decoder = load_keras_model(path + filename + '_decoder.h5')
        prop_pred = load_keras_model(path + filename + '_prop_pred.h5')
        return encoder, decoder, prop_pred

    def update_latent_bounds(self, opt_config):
        m_min = np.min(self.Z_vae, axis=0, keepdims=True) - 0.2  # refined
        m_max = np.max(self.Z_vae, axis=0, keepdims=True) + 0.2
        self.latent_bound = []
        for i in range(self.proj_dim):
            self.latent_bound += [(m_min[0, i].copy(), m_max[0, i].copy())]
        # # uncomment if want to use L-BFGS-B with bounds on the optimization variable
        opt_config['bounds'] = self.latent_bound * self.num_init

        self.latent_grid = np.multiply(np.copy(self.grid), m_max - m_min) + m_min
        return opt_config

    def decode_zopt(self, z_opt):
        try:
            x_opt = self.decoder.predict(z_opt)
        except:
            # tf.reset_default_graph()
            # graph = tf.get_default_graph()
            x_opt = self.decoder.predict(z_opt)

        # x_new = (x_opt.astype(np.float64) * self.X_std) + self.X_mean     # originally mapped to Xnorm
        # x_out = np.clip(x_new, a_min=0., a_max=1.)
        # x_out = norm.cdf((x_new * 2.) - 1.)         # first bring to interval [-1, 1] then evaluate the cdf function
        x_out = norm.cdf(x_opt.astype(np.float64))         # first bring to interval [-1, 1] then evaluate the cdf function
        return x_out


    def run(self, maxiters):
        # load VAE models for a specific objective function
        path = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/chemvae/' + \
               self.dict_args['obj'] + '/'
        encoder, decoder, prop_pred = self.load_vae_models(path)
        self.encoder = encoder
        self.decoder = decoder

        # encode inputs in low-dimensional space
        Z_vae, _ = encoder.predict(self.Xprobit[:, :, None])
        self.Z_vae = Z_vae.astype(np.float64)

        # initialize GP with embedded inputs "Z_vae" and normalized outputs "Ynorm"
        kernel, gp = initialize_models(x=np.copy(self.Z_vae), y=np.copy(self.Ynorm), input_dim=self.proj_dim,
                                       model='GPR', kernel='Matern52', ARD=True)
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')    # check import configuration

        for i in range(maxiters):
            print(i)

            try:
                gpflow.train.ScipyOptimizer().minimize(gp)
            except:
                # if throws error in the optimization of hyper-parameters then set the values to reference
                gp = self.reset_hyps(gp)
            self.hyps.append(self.get_hyps(gp))

            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition(self.loss, gpmodel=gp, **kwargs)   # updated model at each iteration
            opt_config = self.update_latent_bounds(opt_config)
            try:
                z_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)    # check configuration, starting point
            except:
                gp = self.reset_hyps(gp)
                z_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)


            # self.reset_graph()
            # encoder, decoder, prop_pred = self.load_vae_models(path)
            # self.encoder = encoder
            # self.decoder = decoder

            x_tp1 = self.decode_zopt(z_tp1)
            y_tp1 = self.evaluate(x_tp1)

            self.update_data(x_tp1, y_tp1)

            Z_vae, _ = encoder.predict(self.Xprobit[:, :, None])
            self.Z_vae = Z_vae.astype(np.float64)

            self.reset_graph()
            kernel, gp = initialize_models(x=np.copy(self.Z_vae), y=np.copy(self.Ynorm), input_dim=self.proj_dim,
                                           model='GPR', kernel='Matern52', ARD=True)

        return self.data_x, self.data_y, self.hyps, self.log_lik_opt


    def minimize_acquisition(self, acquisition, opt_config):
        acquisition_grid, acq_sum, acq_grad = acquisition(self.latent_grid)
        indices_sorted = np.argsort(acquisition_grid, axis=0)
        x_topk = np.ravel(np.copy(self.latent_grid[indices_sorted[:self.num_init, 0], :]))     # check copy necessary

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