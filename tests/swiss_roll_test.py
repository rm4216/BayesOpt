import numpy as np


# 1D Example
n_intervals = int(10)       # divide the swiss roll into cumulative segments
fractions = np.cumsum(np.ones(shape=[1, n_intervals]) / n_intervals, axis=1)

N = int(500)

z1D_01 = np.linspace(start=0., stop=1., num=N)[:, None]
data_z01 = np.matmul(fractions.transpose(), z1D_01.transpose())

data_z = (data_z01 * 2 + 1) * 3 * np.pi / 2.

data_x = np.multiply(data_z, np.cos(data_z))

data_y = np.multiply(data_z, np.sin(data_z))

data_f = - data_z01 * 5.
np.random.seed(0)
data_noise = np.random.normal(loc=0., scale=0.01, size=data_f.shape)
data_obs = data_f + data_noise

# # plot swiss roll fractions
# import matplotlib.pyplot as plt
# bound_plot_y = (-12.5, 15.5)
# bound_plot_x = (-11., 14.)
# figs, axes = plt.subplots(2, 5, sharey=True)    # !! depends on "n_intervals" !!
# axlist = list(np.ravel(axes))
# for i, (x_i, y_i) in enumerate(zip(data_x, data_y)):
#     axlist[i].scatter(x_i, y_i)
#     axlist[i].set_xlim(bound_plot_x)
#     axlist[i].set_ylim(bound_plot_y)
# plt.savefig('SwissRoll1D' + '.pdf', dpi=None, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format='pdf',
#         transparent=False, bbox_inches='tight', pad_inches=0.1,
#         frameon=None)

x_mean = np.mean(data_x, axis=1)[:, None]
x_std = np.std(data_x, axis=1)[:, None]

y_mean = np.mean(data_y, axis=1)[:, None]
y_std = np.std(data_y, axis=1)[:, None]

Xnorm = (data_x - x_mean) / x_std
Ynorm = (data_y - y_mean) / y_std

for i in range(500):
    for j in range(n_intervals):
        assert (data_x[j, i] - x_mean[j, :])/x_std[j, :] == Xnorm[j, i]


# # plot normalized data
# import matplotlib.pyplot as plt
# bound_plot_x = (-3.5, 2.5)
# bound_plot_y = (-3., 3.5)
# figs, axes = plt.subplots(2, 5, sharey=True)    # !! depends on "n_intervals" !!
# axlist = list(np.ravel(axes))
# for i, (x_i, y_i) in enumerate(zip(Xnorm, Ynorm)):
#     axlist[i].scatter(x_i, y_i)
#     axlist[i].set_xlim(bound_plot_x)
#     axlist[i].set_ylim(bound_plot_y)
# plt.savefig('SwissRollNormalized1D' + '.pdf', dpi=None, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format='pdf',
#         transparent=False, bbox_inches='tight', pad_inches=0.1,
#         frameon=None)


Xnorm_gp = np.stack([Xnorm, Ynorm], axis=0).transpose((1, 2, 0))

obs_mean = np.mean(data_obs, axis=1)[:, None]
obs_std = np.std(data_obs, axis=1)[:, None]

Ynorm_gp = (data_obs - obs_mean) / obs_std

for i in range(500):
    for j in range(n_intervals):
        assert Ynorm_gp[j, i] == (data_obs[j, i] - obs_mean[j, :])/obs_std[j, :]


import random
random.seed(a=123)

class SwissRoll1D(object):
    def __init__(self):
        self.high_dim = int(2)
        self.input_dim = self.high_dim
        self.Npoints = int(5000)

        self.X, self.Y = self.generateSR()

        self.x_bounds = [-11.1, 14.2]

        self.minimizer = np.copy(self.X[-1, :])[None]
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=True)
        assert self.fmin == self.Y[-1, :]
        self.scale = 0.01

    def generateSR(self):
        dataset = np.linspace(start=0., stop=1., num=self.Npoints)[:, None]
        data_z = (dataset * 2 + 1) * 3 * np.pi / 2.
        data_x = np.multiply(data_z, np.cos(data_z))
        data_y = np.multiply(data_z, np.sin(data_z))
        swiss_roll = np.stack([data_x, data_y], axis=0).transpose()
        function_SR = - dataset * 5.
        return swiss_roll[0, :, :], function_SR

    def square_dists(self, xC, X2):
        xCs = np.sum(np.square(xC), axis=1, keepdims=True)          # M x 1
        Xs = np.sum(np.square(X2), axis=1, keepdims=True)           # N x 1
        xXT = np.matmul(xC, np.transpose(X2))                       # M x N
        KL = (xCs + np.transpose(Xs) - 2. * xXT) / 2.               # M x N
        return KL

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if not fulldim:
            x = x * (self.x_bounds[1] - self.x_bounds[0]) + self.x_bounds[0]
        pw_dists2 = self.square_dists(x, self.X)
        indices_min = np.argmin(pw_dists2, axis=1)

        indices_m = np.arange(x.shape[0])
        f_x = np.multiply(self.Y[indices_min, :], np.exp(-pw_dists2[indices_m, indices_min])[:, None])


        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1


sr1D = SwissRoll1D()

Xtest = np.linspace(start=0., stop=1., num=200)
X1, X2 = np.meshgrid(Xtest, Xtest)
Xgrid = np.column_stack([np.ravel(X1), np.ravel(X2)])
Ygrid = sr1D.f(Xgrid, noisy=False, fulldim=False)


# import matplotlib.pyplot as plt
# im = plt.imshow(np.reshape(Ygrid, newshape=X1.shape), vmin=-5., vmax=0., extent=(0, 1, 1, 0))
# im.set_interpolation('bilinear')
# plt.colorbar(im)
# plt.title('Swiss Roll 1D objective')
#
# fig, ax = plt.subplots()
# p = ax.pcolor(X1, X2, np.reshape(Ygrid, newshape=X1.shape), vmin=-5., vmax=0.)
# cb = fig.colorbar(p)
#
# plt.savefig('SwissRollObjective1D' + '.pdf', dpi=None, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format='pdf',
#         transparent=False, bbox_inches='tight', pad_inches=0.1,
#         frameon=None)
# plt.show()





# # Define different proportion for each seed
# import sys, os
# sys.path.insert(0, os.path.join(sys.path[0], '..'))
# from tfbo.utils.import_modules import import_attr
# import argparse
# from collections import OrderedDict
#
# path_to_name_file = 'tfbo/utils/name_file'
# name_attr = import_attr(path_to_name_file, attribute='name_file_start')
#
# path_to_load_save = 'tfbo/utils/load_save'
# save_attr = import_attr(path_to_load_save, attribute='savefile')
#
# names = ['seed', 'obj', 'n_samples']    # the point of this form is to store the order, lists preserve the order
# defaults = ['0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19', 'SwissRoll1D', int(250)]
# types = [str, str, int]
# parser = argparse.ArgumentParser()
# for name_i, default_i, type_i in zip(names, defaults, types):
#     parser.add_argument('--' + name_i, default=default_i, type=type_i)
#
# args = parser.parse_args()
# dict_args = vars(args)
#
# path_to_check_inputs = 'tfbo/utils/check_inputs'
# verify_attr = import_attr(path_to_check_inputs, attribute='verify_dict_inputs')
# verify_attr(dict_args)    # throw error if there are None values
# path_to_transform_inputs = 'tfbo/utils/check_inputs'
# transform_attr = import_attr(path_to_transform_inputs, attribute='transform_inputs')
# dict_args['seed'] = transform_attr(dict_args['seed'])
#
#
# path_to_data = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/datasets/data/' + dict_args['obj'] + '/'
#
# dict_i = OrderedDict([(name_i, dict_args[name_i]) for name_i in names])  # passing an iterable preserves the order
#
# for seed_i in dict_args['seed']:
#     np.random.seed(seed_i)
#     dict_i['seed'] = seed_i
#
#     indices_selection_i = np.random.permutation(np.arange((sr1D.Npoints/20)*(seed_i+1)))[:dict_args['n_samples']]
#
#     _x = sr1D.X[indices_selection_i.astype(int), :][None]
#     _x01 = (_x - sr1D.x_bounds[0]) / (sr1D.x_bounds[1] - sr1D.x_bounds[0])
#     _fx = sr1D.f(_x01[0, :, :], noisy=False, fulldim=False)
#     assert np.max(np.abs(_fx - sr1D.Y[indices_selection_i.astype(int), :])) <= 1e-012
#     _noise = np.random.normal(loc=0., scale=sr1D.scale, size=_fx.shape[0])[:, None]
#     _y = _fx + _noise   # same shape
#
#     x_file, y_file = name_attr(dict_i)
#
#     save_attr(path_to_data + x_file + '.npy', array=_x01)     # str.replace is not in-place modificattion of filename
#     save_attr(path_to_data + y_file + '.npy', array=_y)





# define all seeds for each proportion: proportion = part, i.e. (half roll) part10, part11, ..., part19 (all roll)
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
import argparse
from collections import OrderedDict

path_to_name_file = 'tfbo/utils/name_file'
name_attr = import_attr(path_to_name_file, attribute='name_file_start')

path_to_load_save = 'tfbo/utils/load_save'
save_attr = import_attr(path_to_load_save, attribute='savefile')

names = ['seed', 'obj', 'n_samples']    # the point of this form is to store the order, lists preserve the order
defaults = ['0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19', 'SwissRoll1D', int(50)]
types = [str, str, int]
parser = argparse.ArgumentParser()
for name_i, default_i, type_i in zip(names, defaults, types):
    parser.add_argument('--' + name_i, default=default_i, type=type_i)

args = parser.parse_args()
dict_args = vars(args)

path_to_check_inputs = 'tfbo/utils/check_inputs'
verify_attr = import_attr(path_to_check_inputs, attribute='verify_dict_inputs')
verify_attr(dict_args)    # throw error if there are None values
path_to_transform_inputs = 'tfbo/utils/check_inputs'
transform_attr = import_attr(path_to_transform_inputs, attribute='transform_inputs')
dict_args['seed'] = transform_attr(dict_args['seed'])


parts = [int(10), int(11), int(12), int(13), int(14), int(15), int(16), int(17), int(18), int(19)]

dict_i = OrderedDict([(name_i, dict_args[name_i]) for name_i in names])  # passing an iterable preserves the order

for part_i in parts:
    for seed_i in dict_args['seed']:
        np.random.seed(seed_i)
        dict_i['seed'] = seed_i

        indices_selection_i = np.random.permutation(np.arange((sr1D.Npoints/20)*(part_i+1)))[:dict_args['n_samples']]

        _x = sr1D.X[indices_selection_i.astype(int), :][None]
        _x01 = (_x - sr1D.x_bounds[0]) / (sr1D.x_bounds[1] - sr1D.x_bounds[0])
        _fx = sr1D.f(_x01[0, :, :], noisy=False, fulldim=False)
        assert np.max(np.abs(_fx - sr1D.Y[indices_selection_i.astype(int), :])) <= 1e-012
        _noise = np.random.normal(loc=0., scale=sr1D.scale, size=_fx.shape[0])[:, None]
        _y = _fx + _noise   # same shape

        x_file, y_file = name_attr(dict_i)

        path_to_data = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/datasets/data/' + dict_args[
            'obj'] + '/' + 'part' + str(part_i) + '/'
        save_attr(path_to_data + x_file + '_' + 'part' + str(part_i) + '.npy', array=_x01)     # str.replace is not in-place modificattion of filename
        save_attr(path_to_data + y_file + '_' + 'part' + str(part_i) + '.npy', array=_y)