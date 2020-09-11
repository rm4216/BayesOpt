import gpflow
import numpy as np
import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.models.cov_funcs import ManifoldL2
from tfbo.utils.import_modules import import_attr
from collections import OrderedDict
import tensorflow as tf
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.components.initializations import initialize_acquisition_mo
from tfbo.models.gpr_models import GPR_stable
from tfbo.components.initializations import initialize_models
from tfbo.models.gplvm_models import NN, NN_MOGPR
from tfbo.models.cov_funcs import TreeCoregion
from scipy.optimize import minimize
import argparse
import scipy.io as sio


class manifold_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.initialize_normalization()  # check correct shapes
        self.identity = True


    def initialize_modelM(self):
        k1 = gpflow.kernels.RBF(input_dim=self.proj_dim, ARD=True, active_dims=list(range(self.proj_dim)))
        coreg = gpflow.kernels.Coregion(input_dim=1, output_dim=self.Xnorm.shape[1] + 1,
                                        rank=self.Xnorm.shape[1] + 1, active_dims=[self.proj_dim])
        np.random.seed(23)
        coreg.W = np.random.randn(self.Xnorm.shape[1] + 1, self.Xnorm.shape[1] + 1)

        # values = np.random.normal(loc=0., scale=1., size=[len(self.binary_tree_list)])
        # tree_coregion = TreeCoregion(input_dim=1, output_dim=self.Xnorm.shape[1] + 1,
        #                              indices_tree=self.binary_tree_list, values=values,
        #                              active_dims=[self.proj_dim])

        kc = k1 * coreg
        # kc = k1 * tree_coregion

        nn = NN(dims=[self.Xnorm.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
                name=None)  # otherwise re-initialized at each BO iteration
        # nn = Ort_NN(dims=[self.Xnorm.shape[1], 7, self.proj_dim], N=0, proj_dim=0,
        #         name=None)

        # nn = NN(dims=[self.Xnorm.shape[1], 6, self.proj_dim], N=0, proj_dim=0,
        #         name=None)  # otherwise re-initialized at each BO iteration

        nn_mogp = NN_MOGPR(X=np.copy(self.Xnorm), Y=np.copy(self.Ynorm), kern=kc, nn=nn)
        nn_mogp.likelihood.variance = 1e-06
        nn_mogp.likelihood.variance.trainable = False

        return kc, nn_mogp


    def generate_x(self, x_proj, nn_mogp):
        def decode_i(i):
            decoded_x, var_x = nn_mogp.predict_f(np.hstack([x_proj, np.ones([1, 1]) * i]))
            return decoded_x
        list_x = list(map(decode_i, range(self.input_dim)))
        # x_out = np.clip(np.concatenate(list_x, axis=1), a_min=0., a_max=1.)   # shape
        x_new = (np.concatenate(list_x, axis=1) * self.X_std) + self.X_mean     # originally mapped to Xnorm
        x_out = np.clip(x_new, a_min=0., a_max=1.)
        return x_out


    def run(self, maxiter=20):
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')  # check import configuration
        opt_config['bounds'] = [(0., 1.)] * self.proj_dim * self.num_init
        for j in range(maxiter):
            print('iteration: ', j)
            # initialize model
            self.reset_graph()
            kc, nn_mogp = self.initialize_modelM()
            # train nn and gp hyper-parameters
            # lik0 = nn_mogp.compute_log_likelihood()
            lik0 = nn_mogp.compute_log_likelihood()
            try:
                gpflow.train.ScipyOptimizer().minimize(nn_mogp)
            except:
                self.reset_hyps(nn_mogp)
            lik1 = nn_mogp.compute_log_likelihood()
            Xnn = nn_mogp.nn.np_forward(self.Xnorm)
            self.hyps.append(self.get_hyps(nn_mogp))
            # optimize the acquisition function within 0,1 bounds
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition_mo(loss=self.loss, gpmodel=nn_mogp, input_dim=self.input_dim, **kwargs)
            # acquisition_normalized = lambda x: self.acquisition_norm(acquisition, x, self.X_proj_mean, self.X_proj_std)
            x_proj_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)
            # transform the optimal point into the original input space and clip to 0,1 bound for feasibility
            x_tp1 = self.generate_x(x_proj_tp1, nn_mogp)
            y_tp1 = self.evaluate(x_tp1)
            self.update_data(x_tp1, y_tp1)
            # self.reset_graph()
            # tf.reset_default_graph()
            # print(len(tf.all_variables()))
            # print(len(tf.get_default_graph().get_operations()))
        return self.data_x, self.data_y, self.hyps

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

    def evaluate(self, x_tp1):
        return self.objective.f(x_tp1, noisy=True, fulldim=self.identity)

    def get_hyps(self, gp):
        lengthscales = gp.kern.kernels[0].lengthscales.read_value()
        kern_var = gp.kern.kernels[0].variance.read_value()[None]
        noise_var = gp.likelihood.variance.read_value()[None]
        return np.concatenate([lengthscales, kern_var, noise_var], axis=0)

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


names = ['seed', 'obj', 'opt', 'loss', 'proj_dim', 'input_dim', 'maxiter']
defaults = ['0', 'ProductSines10D', 'manifold_bo', 'Neg_ei', int(10), int(100), int(50)]    # list ints, name_objective, name_optimizer, name loss
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
num_starts = int(50)
shape_starts = [num_starts, dict_args['input_dim']]
Xstart = np.random.uniform(low=0., high=1., size=np.prod(shape_starts)).reshape(shape_starts)

obj_attr = import_attr('datasets/tasks/all_tasks', attribute=dict_args['obj'])
objective = obj_attr()
Ystart = objective.f(Xstart, noisy=True, fulldim=False)
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
        ('hyp_episodes', np.stack(hyps_out, axis=0)[None])
    ])
from tfbo.utils.load_save import save_dictionary
from tfbo.utils.name_file import name_synthetic
path_dict = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/results/'
# filename = 'test_MgpOpt'
filename = name_synthetic(dict_args)
save_dictionary(path_dict + filename + 'seed_' + str(dict_args['seed'][0]) + '.p', dict_out)