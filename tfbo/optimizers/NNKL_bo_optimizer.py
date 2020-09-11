import gpflow
import numpy as np
import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from collections import OrderedDict
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.components.block_diag_initializations import block_diag_initialize_acquisition
from tfbo.components.block_diag_initializations import bloc_diag_initialize_models
from tfbo.models.gplvm_models import NN, Stable_GPR, Ort_NN
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import argparse
import scipy.io as sio
# import random
# random.seed(a=123)

from tfbo.models.cov_funcs import Kstack
from tfbo.models.gplvm_models import NN_MoGPR
from scipy.stats import norm


class NNKL_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        # self.initialize_normalization()  # check correct shapes
        self.initialize_probit()
        self.identity = False
        self.Mo_dim = int(3)
        self.decomposition = [list(np.arange(start=i*self.Mo_dim, stop=(i+1)*self.Mo_dim, step=1)) for i in
                              range(int(np.floor(self.input_dim/self.Mo_dim)))]
        self.Xnn = None
        self.latent_bound = []
        self.latent_grid = []
        self.L = 1.


    def initialize_modelM(self):


        nn = Ort_NN(dims=[self.Xprobit.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
                    name=None)
        # nn = NN(dims=[self.Xprobit.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
        #             name=None)

        k_list, gp_nnjoint = bloc_diag_initialize_models(x=np.copy(self.Xprobit), y=np.copy(self.Ynorm),
                                                 input_dim=self.proj_dim,
                                                 model='joint',
                                                 kernel='Matern52',
                                                 ARD=True,
                                                 nn=nn,
                                                 decomp=self.decomposition)     # last kernel is the Y kernel




        # k_list, gp_nnjoint = initialize_m_models(x=np.copy(self.Xprobit), y=np.copy(self.Ynorm),
        #                                          input_dim=self.proj_dim,
        #                                          model='joint',
        #                                          kernel='Matern52',
        #                                          ARD=True,
        #                                          nn=nn,
        #                                          decomp=self.decomposition)     # last kernel is the Y kernel

        # kern_joint = Kstack(k_list)
        # gp_nnjoint = NN_MoGPR(X=np.copy(self.Xprobit), Y=np.copy(self.Ynorm), kern=kern_joint, nn=nn, Mo_dim=self.Mo_dim)
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

        # mean_new = gp_joint.predict_x(x_proj)
        # mean_new, var_new = gp_joint.predict_x(x_proj)
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


    # def update_latent_bounds(self, opt_config):
    #     m_min = np.clip(np.min(self.Xnn, axis=0, keepdims=True) - 0.2, a_min=0., a_max=1.)  # refined
    #     m_max = np.clip(np.max(self.Xnn, axis=0, keepdims=True) + 0.2, a_min=0., a_max=1.)
    #     self.latent_bound = []
    #     for i in range(self.proj_dim):
    #         self.latent_bound += [(m_min[0, i].copy(), m_max[0, i].copy())]
    #     # # uncomment if want to use L-BFGS-B with bounds on the optimization variable
    #     opt_config['bounds'] = self.latent_bound * self.num_init
    #
    #     self.latent_grid = np.multiply(np.copy(self.grid), m_max - m_min) + m_min
    #     return opt_config


    def learnLipschitz(self, gp_joint):
        rand_ind = np.random.choice(self.Xnn.shape[0], 10, replace=False)
        z_jac_opt = np.copy(self.Xnn[rand_ind, :])
        jac_Dxd_list = []
        for i in range(z_jac_opt.shape[0]):
            jac_list = []
            for j in range(self.input_dim):
                mu_x, jac_ij = gp_joint.jacobian_x(z_jac_opt[i, :][None], j)
                jac_list.append(jac_ij)
            jac_Dxd_i = np.concatenate(jac_list, axis=0)
            jac_Dxd_list.append(jac_Dxd_i)
        jac_NxDxd = np.stack(jac_Dxd_list, axis=0)
        # self.L = np.max(np.abs(jac_NxDxd))
        vec_norms_2 = [np.linalg.norm(jac_i, ord=2) for jac_i in jac_Dxd_list]
        self.L = np.max(vec_norms_2)
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
            # L = 3.
            xC = np.reshape(x, [self.num_init, self.proj_dim])          # M x d
            # xCs = np.sum(np.square(xC), axis=1, keepdims=True)          # M x 1
            # Xs = np.sum(np.square(self.Xnn), axis=1, keepdims=True)     # N x 1
            # xXT = np.matmul(xC, np.transpose(self.Xnn))                 # M x N
            # KL = (xCs + np.transpose(Xs) - 2. * xXT) / 2.               # M x N
            KL = self.KLdist(xC, self.Xnn)                              # M x N
            ind_minimizers = np.argmin(KL, axis=1)                      # (M,)

            x_mean_fast = np.copy(self.Xprobit[ind_minimizers, :])

            # zi_opts = np.copy(self.Xnn[ind_minimizers, :])
            # num_outputs = len(self.decomposition[0])
            # fmean, fvar = gp_joint.predict_x(zi_opts)
            # fmean_reshape = np.reshape(fmean, [len(self.decomposition), num_outputs, zi_opts.shape[0]]).transpose(
            #     (0, 2, 1))
            # x_mean = np.zeros(shape=[zi_opts.shape[0], self.Xprobit.shape[1]])                    # M x D
            # for decomp_kk, kk in zip(self.decomposition, list(range(len(self.decomposition)))):
            #     x_mean[:, decomp_kk] = np.copy(fmean_reshape[kk, :, :])


            # fmean, fvar = gp_joint.predict_x(self.Xnn)
            # fmean_reshape = np.reshape(fmean, [len(self.decomposition), num_outputs, self.Xnn.shape[0]]).transpose((0, 2, 1))
            # x_mean = np.zeros(shape=[self.Xnn.shape[0], self.Xprobit.shape[1]])
            # for decomp_kk, kk in zip(self.decomposition, list(range(len(self.decomposition)))):
            #     x_mean[:, decomp_kk] = fmean_reshape[kk, :, :]
            # errs_x_mean = np.mean(np.abs(x_mean - self.Xprobit), axis=0)

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

            # optimize the acquisition function within 0,1 bounds
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = block_diag_initialize_acquisition(loss=self.loss, gpmodel=gp_nnjoint, **kwargs)
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
# num_starts = int(10)
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