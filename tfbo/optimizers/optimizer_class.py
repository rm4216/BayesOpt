import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
import gpflow
from scipy.stats import norm


class optimizer(object):
    def __init__(self, xy_start, proj_dim, objective, loss):
        self.data_x = np.copy(xy_start[0][0, :, :])
        self.data_y = np.copy(xy_start[1])
        self.data_x_proj = []
        self.input_dim = self.data_x.shape[-1]
        self.proj_dim = proj_dim
        self.objective = objective
        self.loss = loss
        self.hyps = []
        self.num_init = int(100)
        self.grid = self.initialize_grid()

    def initialize_grid(self):
        grid_size = int(5000)
        np.random.seed(123)
        return np.random.uniform(low=0., high=1., size=grid_size*self.proj_dim).reshape([grid_size, self.proj_dim])

    def reset_graph(self):
        tf.reset_default_graph()
        graph = tf.get_default_graph()
        gpflow.reset_default_session(graph=graph)
        # print(len(tf.all_variables()))
        # print(len(tf.get_default_graph().get_operations()))

    def get_hyps(self, gp):
        lengthscales = gp.kern.lengthscales.read_value()
        kern_var = gp.kern.variance.read_value()[None]
        noise_var = gp.likelihood.variance.read_value()[None]
        return np.concatenate([lengthscales, kern_var, noise_var], axis=0)

    # def get_signal_noise(self, gp):
    #     kern_var = gp.kern.variance.read_value()[None]
    #     noise_var = gp.likelihood.variance.read_value()[None]
    #     return np.concatenate([kern_var, noise_var], axis=0)

    def reset_hyps(self, gp):
        gp.kern.lengthscales = np.ones(shape=[self.proj_dim])  # check shape, check transformation hyps
        gp.kern.variance = 1.
        gp.likelihood.variance = 1.
        # gp.read_trainables()
        return gp

    def assign_hyps(self, gp, hyps_array):
        gp.kern.lengthscales = hyps_array[:-2]
        gp.kern.variance = hyps_array[-2]
        gp.likelihood.variance = hyps_array[-1]
        return gp

    def minimize_acquisition(self, acquisition, opt_config):
        acquisition_grid, acq_sum, acq_grad = acquisition(self.grid)
        indices_sorted = np.argsort(acquisition_grid, axis=0)
        x_topk = np.ravel(np.copy(self.grid[indices_sorted[:self.num_init, 0], :]))     # check copy necessary

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

    def initialize_normalization(self):
        self.X_mean = np.mean(self.data_x, axis=0, keepdims=True)
        self.X_std = np.std(self.data_x, axis=0, keepdims=True)

        self.Y_mean = np.mean(self.data_y, axis=0, keepdims=True)
        self.Y_std = np.std(self.data_y, axis=0, keepdims=True)

        self.Xnorm = (self.data_x - self.X_mean) / self.X_std
        self.Ynorm = (self.data_y - self.Y_mean) / self.Y_std   # check all shapes

        def test_normalization(input_norm):
            xm_test = np.mean(input_norm, axis=0, keepdims=True)
            xst_test = np.std(input_norm, axis=0, keepdims=True) - np.ones([1, input_norm.shape[1]])
            boolm_list = [np.abs(xm_i[0]) <= 1e-02 for xm_i in list(xm_test.transpose())]
            bools_list = [np.abs(xs_i[0]) <= 1e-02 for xs_i in list(xst_test.transpose())]
            assert all(boolm_list) and all(bools_list)
        test_normalization(self.Xnorm)
        test_normalization(self.Ynorm)

    def initialize_probit(self):
        self.Xprobit = norm.ppf(self.data_x)    # probit function to compute inverse of Gaussian cdf [0, 1] -> [-inf inf]
        self.Xprobit = np.clip(self.Xprobit, a_min=-37.5, a_max=37.5)
        # assert np.max(np.abs(norm.cdf(self.Xprobit) - self.data_x)) < 1e-09

        self.Y_mean = np.mean(self.data_y, axis=0, keepdims=True)
        self.Y_std = np.std(self.data_y, axis=0, keepdims=True)

        self.Ynorm = (self.data_y - self.Y_mean) / self.Y_std   # check all shapes

        def test_normalization(input_norm):
            xm_test = np.mean(input_norm, axis=0, keepdims=True)
            xst_test = np.std(input_norm, axis=0, keepdims=True) - np.ones([1, input_norm.shape[1]])
            boolm_list = [np.abs(xm_i[0]) <= 1e-02 for xm_i in list(xm_test.transpose())]
            bools_list = [np.abs(xs_i[0]) <= 1e-02 for xs_i in list(xst_test.transpose())]
            assert all(boolm_list) and all(bools_list)
        test_normalization(self.Ynorm)

    def acquisition_norm(self, acquisition, x, X_proj_mean, X_proj_std):
        # wrapper to normalize the input
        N = x.shape[0]
        X_proj_mean_rep = np.tile(X_proj_mean, reps=[N, 1])
        X_proj_std_rep = np.tile(X_proj_std, reps=[N, 1])
        xnorm = (x - X_proj_mean_rep) / X_proj_std_rep
        acq_norm, acq_sum, acq_grad = acquisition(xnorm)
        return acq_norm, acq_sum, acq_grad

    def sigmoid(self, sample):
        if len(sample.shape) == 1:
            sample = sample[None]
        sample = np.clip(sample, a_min=-np.log(1e09), a_max=1e09)
        return np.divide(1., 1. + np.exp(-sample))

    def logit(self, x_input):
        if len(x_input.shape) == 1:
            x_input = x_input[None]
        x_input = np.clip(x_input, a_min=1e-06, a_max=1. - 1e-06)
        return - np.log(np.divide(1., x_input) - 1.)