import numpy as np
import gpflow
from tfbo.utils.import_modules import import_attr
from tfbo.components.initializations import initialize_models, initialize_acquisition
from scipy.optimize import minimize
from tfbo.optimizers.optimizer_class import optimizer
from scipy.stats import norm


class rembo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, seed, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        np.random.seed(seed)
        self.seed = seed
        self.identity = self.set_identity_flag()
        self.x_bounds = self.initialize_x_bounds()          # bounds original space
        self.proj_bounds = self.initialize_proj_bounds()    # bounds embedded space

        self.A = self.initialize_A(identity=self.identity)

        self.data_x_proj = self.inverse_linear_map(self.data_x)
        self.grid = self.initialize_grid_rembo()
        self.initialize_normalization()
        self.log_lik_opt = []
        self.hyps_opt = []

    def set_identity_flag(self):
        if self.input_dim == self.proj_dim:
            return True
        else:
            return False

    def initialize_grid_rembo(self):
        # grid points for the optimization of the acquisition function are in the interval [-sqrt(proj_dim), sqrt(proj_dim)]
        grid_size = int(5000)
        return np.random.uniform(low=self.proj_bounds[0][0], high=self.proj_bounds[1][0],
                                 size=grid_size*self.proj_dim).reshape([grid_size, self.proj_dim])

    def initialize_x_bounds(self):
        x_bounds = \
            [
                - np.ones(shape=[self.input_dim]),
                np.ones(shape=[self.input_dim])
            ]
        return x_bounds     # rank 1    necessary for clipping values

    def initialize_proj_bounds(self):   # check correctness paper
        const = 5. / np.sqrt(self.proj_dim)
        # const = 1. / np.log(self.proj_dim)
        proj_bounds = \
            [
                - np.ones(shape=[self.proj_dim]) * const,
                np.ones(shape=[self.proj_dim]) * const
            ]
        return proj_bounds  # rank 1

    def initialize_A(self, identity=False):
        '''
        We define a random matrix that is a matrix that has entries sampled from standard Gaussian N(\mu=0, \sigma=1)
        :param identity:
        :return:
        '''
        if identity:
            assert self.input_dim == self.proj_dim
            return np.eye(self.input_dim)

        if self.input_dim == 12:
            path_load = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/rembo/demos/ElectronSphere6np/A_orth_matrices/'
            from scipy.io import loadmat
            try:
                path_orth_A = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/rembo/demos/ElectronShpere6np/A_orth_matrices/A_orth_seed=' + str(self.seed + 1) + '.mat'
                mat_dict = loadmat(path_orth_A)
                # mat_dict = loadmat(path_load + 'A_orth_seed=' + str(self.seed + 1) + '.mat')
            except:
                print('Loading from server machines')
                # path_load = '/homes/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/rembo/demos/ElectronSphere6np/A_orth_matrices/'
                path_orth_A = '/homes/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/rembo/demos/ElectronShpere6np/A_orth_matrices/A_orth_seed=' + str(
                    self.seed + 1) + '.mat'
                mat_dict = loadmat(path_orth_A)
                # mat_dict = loadmat(path_load.replace('home', 'homes') + 'A_orth_seed=' + str(self.seed + 1) + '.mat')
            q = np.copy(np.array(mat_dict['A12x12'][:, 0:self.proj_dim]))
        else:
            A = np.random.normal(loc=0., scale=1., size=self.input_dim*self.proj_dim).\
                reshape([self.input_dim, self.proj_dim])
            q, r = np.linalg.qr(A, mode='reduced')      # q has shape (self.input_dim, self.proj_dim)
            assert q.shape[0] == A.shape[0] and q.shape[1] == A.shape[1]
        # return A
        return q

    def in_bounds(self, x_proj_minimizer):
        def in_bound_i(components, index, self):
            return components[0, index] <= self.proj_bounds[1][index] and components[0, index] >= self.proj_bounds[0][index]
        in_bounds = lambda index: in_bound_i(components=x_proj_minimizer, index=index, self=self)
        list_bools = list(map(in_bounds, list(range(self.proj_dim))))    # a bool for each dimension in the original space
        return all(list_bools)

    def linear_map(self, x_proj):                   # Ax_proj=x  <=>  embedd -> original_space
        assert x_proj.shape[1] == self.proj_dim     # x_proj.shape:  N x d (embedded space)
        # N = x_proj.shape[0]
        # xm11 = np.clip(np.matmul(x_proj, np.transpose(self.A)), a_min=np.tile(self.x_bounds[0], reps=[N, 1]),
        #                a_max=np.tile(self.x_bounds[1], reps=[N, 1]))  # a_min, a_max have shapes [N x D] check that
        xm11 = np.matmul(x_proj, np.transpose(self.A))          # ambient space [-1, 1]
        # # x01 = (xm11 + 1.) / 2.                            # ambient space [ 0, 1]
        # xsoftmax = norm.cdf(xm11)                           # ambient space [-1, 1] -> push through activation function Gaussian.cdf
        # return xsoftmax    # x in [0, 1]
        return np.clip((xm11 + 1.) / 2., a_min=0., a_max=1.)    # ambient space in [-1, 1] -> ambient space in [0, 1] and clip

    def inverse_linear_map(self, x):                # x_proj=pinv(A)x  <=>  original_space -> embedd
        # map points "x" in the ambient space to "point x_proj" in embedded space which have the same objective value before clipping
        # the inputs x_proj are in the interval [-1, 1]
        assert x.shape[1] == self.input_dim
        N = x.shape[0]
        if self.identity:
            return (x * 2.) - 1.
        else:
            xm11 = (x * 2.) - 1.    # x in [0,1] to x in [-1, 1]
            return np.clip(np.matmul(xm11, np.transpose(np.linalg.pinv(self.A))),
                           a_min=np.tile(self.proj_bounds[0], reps=[N, 1]), a_max=np.tile(self.proj_bounds[1], reps=[N, 1]))

    def update_data(self, x_proj, y):
        x = self.linear_map(x_proj)
        self.data_x_proj = np.concatenate([self.data_x_proj, x_proj], axis=0)
        self.data_x = np.concatenate([self.data_x, x], axis=0)     # check shapes
        self.data_y = np.concatenate([self.data_y, y], axis=0)
        self.initialize_normalization()

    def evaluate(self, x_proj):
        assert x_proj.shape[1] == self.proj_dim
        x = self.linear_map(x_proj)
        y = self.objective.f(x, noisy=True, fulldim=self.identity)
        return y

    # def fit_gp(self, gp):
    #     self.hyps_opt = []
    #     signal_variance = np.random.uniform(low=0., high=10, size=10)[:, None]
    #     noise_variance = signal_variance * 0.01
    #     log_lik = []
    #     for signal_i, noise_i in zip(list(signal_variance), list(noise_variance)):
    #         self.reset_hyps(gp)
    #         gp.kern.variance = signal_i[0]
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
        kernel, gp = initialize_models(x=np.copy(self.X_proj_norm), y=np.copy(self.Ynorm), input_dim=self.proj_dim,
                                       model='GPR', kernel='Matern52', ARD=True)
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')    # check import configuration
        opt_config['bounds'] = [(self.proj_bounds[0][i], self.proj_bounds[1][i]) for i in range(self.proj_dim)] * \
                               self.num_init
        # opt_hyp = import_attr('tfbo/configurations', attribute='hyp_opt')
        # # var_list = gp.
        # opt_hyp['var_to_bounds'] = [(np.log(np.exp(1e-04) - 1.), np.log(np.exp(1e04) - 1.))] + \
        #                            [(np.log(np.exp(1e-06) - 1.), np.log(np.exp(1e06) - 1.))] * self.proj_dim + \
        #                            [(np.log(np.exp(1e-08) - 1.), np.log(np.exp(1e08) - 1.))]   # order of hyps in list

        for i in range(maxiters):
            print(i)
            # gp = self.fit_gp(gp)
            try:
                gpflow.train.ScipyOptimizer().minimize(gp)
            except:
                # if throws error in the optimization of hyper-parameters then set the values to reference
                gp = self.reset_hyps(gp)
                # gpflow.train.ScipyOptimizer().minimize(gp)
            self.hyps.append(self.get_hyps(gp))

            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition(self.loss, gpmodel=gp, **kwargs)   # updated model at each iteration
            # def acquisition_norm(acquisition, x, X_proj_mean, X_proj_std):
            #     # wrapper to normalize the input
            #     N = x.shape[0]
            #     X_proj_mean_rep = np.tile(X_proj_mean, reps=[N, 1])
            #     X_proj_std_rep = np.tile(X_proj_std, reps=[N, 1])
            #     xnorm = (x - X_proj_mean_rep) / X_proj_std_rep
            #     acq_norm, acq_sum, acq_grad = acquisition(xnorm)
            #     return acq_norm, acq_sum, acq_grad
            acquisition_normalized = lambda x: self.acquisition_norm(acquisition, x, self.X_proj_mean, self.X_proj_std)
            # acquisition_normalized = lambda x: acquisition_norm(acquisition, x, self.X_proj_mean, self.X_proj_std)  # check broadcasting

            try:
                x_tp1, acq_tp1 = self.minimize_acquisition(acquisition_normalized, opt_config)    # check configuration, starting point
            except:
                np.save('Xcrash_rembo', gp.X.read_value())
                np.save('Ycrash_rembo', gp.Y.read_value())
                np.save('hyps_crash', self.get_hyps(gp))
                gp = self.reset_hyps(gp)
                x_tp1, acq_tp1 = self.minimize_acquisition(acquisition_normalized, opt_config)
            y_tp1 = self.evaluate(x_tp1)

            self.update_data(x_tp1, y_tp1)
            self.reset_graph()
            kernel, gp = initialize_models(x=np.copy(self.X_proj_norm), y=np.copy(self.Ynorm), input_dim=self.proj_dim,
                                           model='GPR', kernel='Matern52', ARD=True)
            # gp = self.update_model(gp)
        return self.data_x, self.data_y, self.hyps, self.log_lik_opt

    def initialize_normalization(self):
        self.X_mean = np.mean(self.data_x, axis=0, keepdims=True)
        self.X_std = np.std(self.data_x, axis=0, keepdims=True)

        self.X_proj_mean = np.mean(self.data_x_proj, axis=0, keepdims=True)
        self.X_proj_std = np.std(self.data_x_proj, axis=0, keepdims=True)

        self.Y_mean = np.mean(self.data_y, axis=0, keepdims=True)
        self.Y_std = np.std(self.data_y, axis=0, keepdims=True)

        self.Xnorm = (self.data_x - self.X_mean) / self.X_std
        self.Ynorm = (self.data_y - self.Y_mean) / self.Y_std
        self.X_proj_norm = (self.data_x_proj - self.X_proj_mean) / self.X_proj_std  # check shapes

        def test_normalization(input_norm):
            xm_test = np.mean(input_norm, axis=0, keepdims=True)
            xst_test = np.std(input_norm, axis=0, keepdims=True) - np.ones([1, input_norm.shape[1]])
            boolm_list = [np.abs(xm_i[0]) <= 1e-02 for xm_i in list(xm_test.transpose())]
            bools_list = [np.abs(xs_i[0]) <= 1e-02 for xs_i in list(xst_test.transpose())]
            assert all(boolm_list) and all(bools_list)
        test_normalization(self.Xnorm)
        test_normalization(self.Ynorm)
        test_normalization(self.X_proj_norm)

    def update_model(self, gp):
        gp.X = self.X_proj_norm
        gp.Y = self.Ynorm
        return gp