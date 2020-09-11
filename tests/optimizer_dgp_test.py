import gpflow
import numpy as np
import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from collections import OrderedDict
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.components.initializations import initialize_acquisition_dgp
sys.path.insert(0, "/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Doubly_Stochastic_DGP")
sys.path.insert(0, "/homes/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Doubly_Stochastic_DGP")
# from doubly_stochastic_dgp.dgp import DGP
from doubly_stochastic_dgp.dgpMo import DGP
import tensorflow as tf
from gpflow.training import AdamOptimizer, NatGradOptimizer
from gpflow.likelihoods import Gaussian
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
import argparse
from gpflow.actions import Loop


class manifold_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.initialize_normalization()  # check correct shapes
        self.identity = True
        self.latent_grid = None
        self.latent_bound = []
        # self.x_proj_history = []


    def initialize_modelM(self):
        M = np.minimum(int(100), self.Xnorm.shape[0])
        Z_M = kmeans2(np.copy(self.Xnorm), M, minit='points')[0]

        # X -> z -> X : GP0(X -> z), GP1(z -> XY) Deep GP autoencoder
        dims = [self.Xnorm.shape[1], int(2)]

        # define a list of kernels for the deep GPs
        kernels = []
        for dim_i in dims:
            kernels.append(gpflow.kernels.RBF(input_dim=dim_i, ARD=True, lengthscales=np.ones(shape=[dim_i]) * 0.2))    # not relevant any more
            # + gpflow.kernels.Matern32(input_dim=dim_i, ARD=True, lengthscales=np.ones(shape=[dim_i]) * 0.2))

        # add white noise
        for kernel_i, dim_i in zip(kernels[:-1], dims[:-1]):
            kernel_i += gpflow.kernels.White(input_dim=dim_i, variance=1e-05)

        # define the deep GP model
        XYnorm = np.concatenate([self.Xnorm, self.Ynorm], axis=1)
        dgp_model = DGP(X=np.copy(self.Xnorm), Y=np.copy(XYnorm),
                        Z=Z_M, kernels=kernels, likelihood=Gaussian(),
                        num_samples=5,
                        minibatch_size=None)

        # start the inner layers almost deterministically
        for layer in dgp_model.layers[:-1]:
            layer.q_sqrt = layer.q_sqrt.value * 1e-5

        # # start with small lengthscales
        # for layer_i, dim_i in zip(dgp_model.layers, dims):
        #     layer.kern.lengthscales = np.ones(shape=[dim_i]) * 0.2

        dgp_model.likelihood.likelihood.variance = 0.01

        return kernels, dgp_model


    def train_model(self, dgp_model):


        ng_vars = [[dgp_model.layers[-1].q_mu, dgp_model.layers[-1].q_sqrt]]
        for v in ng_vars[0]:
            v.set_trainable(False)
        ng_action = NatGradOptimizer(gamma=0.1).make_optimize_action(dgp_model, var_list=ng_vars)
        adam_action = AdamOptimizer(0.01).make_optimize_action(dgp_model)

        iterations = 10000
        try:
            Loop([ng_action, adam_action], stop=iterations)()
        except:
            print('Failure of Cholesky in Nat Gradient')

        # sess = dgp_model.enquire_session()
        #
        # gamma_start = 1e-2
        # gamma_max = 1e-1
        # gamma_step = 1e-2
        #
        # gamma = tf.Variable(gamma_start, dtype=tf.float64)
        # gamma_incremented = tf.where(tf.less(gamma, gamma_max), gamma + gamma_step, gamma_max)
        #
        # op_ng = NatGradOptimizer(gamma).make_optimize_tensor(dgp_model, var_list=[[dgp_model.layers[-1].q_mu,
        #                                                                            dgp_model.layers[-1].q_sqrt]])
        # op_adam = AdamOptimizer(0.001).make_optimize_tensor(dgp_model)
        # op_increment_gamma = tf.assign(gamma, gamma_incremented)
        #
        # gamma_fallback = 1e-1  # we'll reduce by this factor if there's a cholesky failure
        # op_fallback_gamma = tf.assign(gamma, gamma * gamma_fallback)
        #
        # sess.run(tf.variables_initializer([gamma]))
        #
        # iterations = 10000
        # for it in range(iterations):
        #     try:
        #         sess.run(op_ng)
        #         sess.run(op_increment_gamma)
        #     except tf.errors.InvalidArgumentError:
        #         g = sess.run(gamma)
        #         print('gamma = {} on iteration {} is too big! Falling back to {}'.format(it, g, g * gamma_fallback))
        #         sess.run(op_fallback_gamma)
        #
        #     sess.run(op_adam)
        #
        #     if it % 1000 == 0:
        #         print('{} gamma={:.4f} ELBO={:.4f}'.format(it, *sess.run([gamma, dgp_model.likelihood_tensor])))
        #
        # dgp_model.anchor(sess)
        # # print(len(tf.all_variables()))
        # # print(len(tf.get_default_graph().get_operations()))
        sess = dgp_model.enquire_session()
        dgp_model.anchor(sess)
        print('ELBO={:.4f}'.format(*sess.run([dgp_model.likelihood_tensor])))
        return dgp_model


    def update_latent_bounds(self, dgp_model, opt_config):
        samples, mean, var = dgp_model.predict_all_layers(self.Xnorm, 1)
        m_min = np.min(mean[0][0, :, :], axis=0, keepdims=True)  # refined
        m_max = np.max(mean[0][0, :, :], axis=0, keepdims=True)
        self.latent_bound = []
        for i in range(self.proj_dim):
            self.latent_bound += [(m_min[0, i].copy(), m_max[0, i].copy())]
        # # uncomment if want to use L-BFGS-B with bounds on the optimization variable
        opt_config['bounds'] = self.latent_bound * self.num_init

        self.latent_grid = np.multiply(np.copy(self.grid), m_max - m_min) + m_min
        return opt_config


    def generate_x(self, x_proj, dgp):
        sample, x, var = dgp.predict_last_layer(x_proj, 1)
        x_new = (x[0, :, :self.input_dim].copy() * self.X_std) + self.X_mean     # originally mapped to Xnorm
        x_out = np.clip(x_new, a_min=0., a_max=1.)
        return x_out


    def run(self, maxiter=20):
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')  # check import configuration
        # opt_config = import_attr('tfbo/configurations', attribute='bfgs_opt')  # check import configuration
        for j in range(maxiter):
            print('iteration: ', j)
            # initialize model
            self.reset_graph()
            _, dgp_model = self.initialize_modelM()
            # train deep gp hyper-parameters
            dgp_model = self.train_model(dgp_model)
            self.hyps.append(self.get_hyps(dgp_model))
            # initialize acquisition function
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition_dgp(loss=self.loss, gpmodel=dgp_model, num_samples=int(1), **kwargs)
            # only latent grid is modified, configurations of BFGS are the same
            opt_config = self.update_latent_bounds(dgp_model, opt_config)
            x_proj_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)
            # transform the optimal point into the original input space and clip to 0,1 bound for feasibility
            x_tp1 = self.generate_x(x_proj_tp1, dgp_model)
            y_tp1 = self.evaluate(x_tp1)
            self.update_data(x_tp1, y_tp1)
            # self.reset_graph()
            # tf.reset_default_graph()
            # print(len(tf.all_variables()))
            # print(len(tf.get_default_graph().get_operations()))
        return self.data_x, self.data_y, self.hyps

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

    def get_hyps(self, dgp):
        def lengthscales(layer_i):
            return np.concatenate([layer_i.kern.kernels[i].kernels[0].lengthscales.read_value() for i in range(layer_i.num_outputs)], axis=0)
        lengthscales_list = list(map(lengthscales, dgp.layers))    # gp.kern.kernels[0].lengthscales.read_value()
        l = np.concatenate(lengthscales_list, axis=0)
        def amplitude(layer_i):
            return np.concatenate([layer_i.kern.kernels[i].kernels[0].variance.read_value()[None] for i in range(layer_i.num_outputs)], axis=0)
        kv = list(map(amplitude, dgp.layers))                      # gp.kern.kernels[0].variance.read_value()[None]
        kvar = np.concatenate(kv, axis=0)
        noise_var = dgp.likelihood.likelihood.variance.read_value()[None]
        return np.concatenate([l, kvar, noise_var], axis=0)

    def update_data(self, x_new, y_new):
        self.data_x = np.concatenate([self.data_x, x_new], axis=0)     # check shapes
        self.data_y = np.concatenate([self.data_y, y_new], axis=0)
        self.initialize_normalization()

    def reset_hyps(self, gp):
        gp.kern.lengthscales = np.ones(shape=[self.proj_dim])  # check shape, check transformation hyps
        gp.kern.variance = 1.
        gp.likelihood.variance = 1.
        # gp.read_trainables()
        return gp


# dict_args = OrderedDict(
#     [
#         ('seed', int(0)),
#         ('obj', 'Branin2D'),
#         ('opt', 'manifold_bo'),
#         ('loss', 'Neg_ei'),
#         ('proj_dim', int(1)),
#         ('input_dim', int(2))
#     ])

# dictionary optimization inputs
names = ['seed', 'obj', 'opt', 'loss', 'proj_dim', 'input_dim', 'maxiter']
defaults = ['0', 'Hartmann6D', 'dgpMo_bo', 'Neg_ei', int(2), int(6), int(10)]    # list ints, name_objective, name_optimizer, name loss
types = [str, str, str, str, int, int, int, float]
parser = argparse.ArgumentParser(description='Input: list numbers, name_objective, name_optimizer, name loss, proj_dim, input_dim, maxiters')
for name_i, default_i, type_i in zip(names, defaults, types):
    parser.add_argument('--' + name_i, default=default_i, type=type_i)
args = parser.parse_args()
dict_args = vars(args)
print(dict_args)
# check inputs
verify_attr = import_attr('tfbo/utils/check_inputs', attribute='verify_dict_inputs')
verify_attr(dict_args)
string_to_int_attr = import_attr('tfbo/utils/check_inputs', attribute='transform_inputs')
dict_args['seed'] = string_to_int_attr(dict_args['seed'])


np.random.seed(dict_args['seed'])

# Generate starting data
num_starts = int(10)
shape_starts = [num_starts, dict_args['input_dim']]
Xstart = np.random.uniform(low=0., high=1., size=np.prod(shape_starts)).reshape(shape_starts)

obj_attr = import_attr('datasets/tasks/all_tasks', attribute=dict_args['obj'])
objective = obj_attr()
Ystart = objective.f(Xstart, noisy=True, fulldim=True)
xy_start = [Xstart[None], Ystart]

optimizerM = manifold_bo_optimizer(xy_start,
                                   proj_dim=dict_args['proj_dim'],
                                   objective=objective,
                                   loss=dict_args['loss'])
x_out, y_out, hyps_out = optimizerM.run(maxiter=dict_args['maxiter'])

dict_out = OrderedDict(
    [
        ('Xepisodes', x_out[None]),
        ('Yepisodes', y_out),
        # ('Xproj_episodes', x_projs),
        ('hyp_episodes', np.stack(hyps_out, axis=0)[None])
    ])
from tfbo.utils.load_save import save_dictionary
from tfbo.utils.name_file import name_synthetic
path_dict = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/results/'
# filename = 'test_MgpOpt'
filename = name_synthetic(dict_args)
save_dictionary(path_dict + filename + 'seed_' + str(dict_args['seed'][0]) + '.p', dict_out)