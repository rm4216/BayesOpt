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
import argparse
import scipy.io as sio


class NNdMo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.initialize_normalization()  # check correct shapes
        self.identity = False
        self.Mo_dim = int(3)
        self.decomposition = [list(np.arange(start=i*self.Mo_dim, stop=(i+1)*self.Mo_dim, step=1)) for i in
                              range(int(np.floor(self.input_dim/self.Mo_dim)))]
        self.Xnn = None


    def initialize_modelM(self):


        nn = Ort_NN(dims=[self.Xnorm.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
                    name=None)

        kernm, gpm = initialize_m_models(x=np.copy(self.Xnorm), y=np.copy(self.Ynorm),
                                         input_dim=self.proj_dim,
                                         model='encoder',
                                         kernel='Matern52',
                                         ARD=True,
                                         nn=nn,
                                         decomp=None)

        gpm.likelihood.variance = 0.01

        return kernm, gpm


    def generate_x(self, x_proj, gp_list):

        for gp_i, i in zip(gp_list, list(range(len(gp_list)))):
            print('Training decoder block: ', i)
            try:
                gpflow.train.ScipyOptimizer().minimize(gp_i)
            except:
                self.reset_hyps_mo(gp_i)

        num_outputs = len(self.decomposition[0])
        x_input = np.vstack([np.hstack([x_proj, np.ones([1, 1]) * j]) for j in range(num_outputs)])
        # xnn_input = np.vstack(
        #     [np.hstack([self.Xnn, np.ones(shape=[self.Xnn.shape[0], 1]) * j]) for j in range(num_outputs)])
        def decode_i(gp_i):
            decoded_x, var_x = gp_i.predict_f(x_input)
            # recon_x, s2_x = gp_i.predict_f(xnn_input)
            # x_test = np.reshape(recon_x, [num_outputs, self.Xnn.shape[0]]).transpose()    # for comparison with Xnorm in batches of columns
            return decoded_x
        list_x = list(map(decode_i, gp_list))

        x_concat = np.zeros(shape=[1, self.input_dim])
        for x_i, decomp_i in zip(list_x, self.decomposition):
            x_concat[:, decomp_i] = x_i[:, 0]
        x_new = (x_concat * self.X_std) + self.X_mean     # originally mapped to Xnorm
        x_out = np.clip(x_new, a_min=0., a_max=1.)
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
        opt_config['bounds'] = [(0., 1.)] * self.proj_dim * self.num_init
        for j in range(maxiters):
            print('iteration: ', j)
            # initialize model
            self.reset_graph()
            km, gpm = self.initialize_modelM()
            # train nn and gp hyper-parameters
            try:
                gpflow.train.ScipyOptimizer().minimize(gpm)
            except:
                self.reset_hyps(gpm)

            Xnn = gpm.nn.np_forward(self.Xnorm)
            self.Xnn = Xnn
            self.hyps.append(self.get_hyps(gpm))

            gp_acq = Stable_GPR(X=np.copy(Xnn), Y=np.copy(self.Ynorm), kern=km)
            # gp_acq.likelihood.variance = gpm.likelihood.variance.read_value()

            # optimize the acquisition function within 0,1 bounds
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition(loss=self.loss, gpmodel=gp_acq, **kwargs)
            opt_config = self.update_latent_bounds(opt_config)
            x_proj_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)

            k_list, gp_list = initialize_m_models(x=np.copy(Xnn), y=np.copy(self.Xnorm),
                                                  input_dim=self.proj_dim,
                                                  model='decoder',
                                                  kernel='Matern52',
                                                  ARD=True,
                                                  nn=None,
                                                  decomp=self.decomposition)
            # transform the optimal point into the original input space and clip to 0,1 bound for feasibility
            x_tp1 = self.generate_x(x_proj_tp1, gp_list)

            y_tp1 = self.evaluate(x_tp1)
            self.update_data(x_tp1, y_tp1)
        lik = []
        return self.data_x, self.data_y, self.hyps, lik

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

    def evaluate(self, x_tp1):
        return self.objective.f(x_tp1, noisy=True, fulldim=self.identity)

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
        lengthscales = gp.kern.lengthscales.read_value()
        kern_var = gp.kern.variance.read_value()[None]
        noise_var = gp.likelihood.variance.read_value()[None]
        return np.concatenate([lengthscales, kern_var, noise_var], axis=0)


# # dict_args = OrderedDict(
# #     [
# #         ('seed', int(0)),
# #         ('obj', 'Branin2D'),
# #         ('opt', 'manifold_bo'),
# #         ('loss', 'Neg_ei'),
# #         ('proj_dim', int(1)),
# #         ('input_dim', int(2))
# #     ])
#
# # dictionary optimization inputs
# names = ['seed', 'obj', 'opt', 'loss', 'proj_dim', 'input_dim', 'maxiters']
# defaults = ['0', 'Hartmann6D', 'NNd_bo', 'Neg_ei', int(6), int(60), int(5)]    # list ints, name_objective, name_optimizer, name loss
# types = [str, str, str, str, int, int, int, float]
# parser = argparse.ArgumentParser(description='Input: list numbers, name_objective, name_optimizer, name loss, proj_dim, input_dim, maxiters')
# for name_i, default_i, type_i in zip(names, defaults, types):
#     parser.add_argument('--' + name_i, default=default_i, type=type_i)
# args = parser.parse_args()
# dict_args = vars(args)
# print(dict_args)
# # check inputs
# verify_attr = import_attr('tfbo/utils/check_inputs', attribute='verify_dict_inputs')
# verify_attr(dict_args)
# string_to_int_attr = import_attr('tfbo/utils/check_inputs', attribute='transform_inputs')
# dict_args['seed'] = string_to_int_attr(dict_args['seed'])
#
#
#
# np.random.seed(dict_args['seed'])
#
# # Generate starting data
# num_starts = int(200)
# shape_starts = [num_starts, dict_args['input_dim']]
# Xstart = np.random.uniform(low=0., high=1., size=np.prod(shape_starts)).reshape(shape_starts)
#
# obj_attr = import_attr('datasets/tasks/all_tasks', attribute=dict_args['obj'])
# objective = obj_attr()
# Ystart = objective.f(Xstart, noisy=True, fulldim=False)
# xy_start = [Xstart[None], Ystart]
#
# optimizerM = manifold_bo_optimizer(xy_start,
#                                    proj_dim=dict_args['proj_dim'],
#                                    objective=objective,
#                                    loss=dict_args['loss'])
# x_out, y_out, hyps_out = optimizerM.run(maxiters=dict_args['maxiters'])
#
# dict_out = OrderedDict(
#     [
#         ('Xepisodes', x_out[None]),
#         ('Yepisodes', y_out),
#         # ('Xproj_episodes', x_projs),
#         ('hyp_episodes', np.stack(hyps_out, axis=0)[None])
#     ])
# from tfbo.utils.load_save import save_dictionary
# from tfbo.utils.name_file import name_synthetic
# path_dict = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/results/'
# # filename = 'test_MgpOpt'
# filename = name_synthetic(dict_args)
# save_dictionary(path_dict + filename + 'seed_' + str(dict_args['seed'][0]) + '.p', dict_out)