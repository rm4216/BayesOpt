import numpy as np
import argparse
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from collections import OrderedDict


names = ['seed', 'obj', 'n_samples']    # the point of this form is to store the order, lists preserve the order
defaults = ['0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19', 'ProductSinesLinear10D', int(10)]
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

path_to_tasks = 'datasets/tasks/all_tasks'
obj_attr = import_attr(path_to_tasks, attribute=dict_args['obj'])
objective = obj_attr()
high_dim = objective.high_dim
shape_inputs = [1, dict_args['n_samples'], high_dim]

path_to_name_file = 'tfbo/utils/name_file'
name_attr = import_attr(path_to_name_file, attribute='name_file_start')

path_to_load_save = 'tfbo/utils/load_save'
save_attr = import_attr(path_to_load_save, attribute='savefile')

path_to_data = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/datasets/data/' + dict_args['obj'] + '/'

dict_i = OrderedDict([(name_i, dict_args[name_i]) for name_i in names])  # passing an iterable preserves the order

for seed_i in dict_args['seed']:
    np.random.seed(seed_i)
    dict_i['seed'] = seed_i

    _x = np.random.uniform(low=0., high=1., size=np.prod(shape_inputs)).reshape(shape_inputs)
    _fx = objective.f(_x[0, :, :], noisy=False, fulldim=False)
    _noise = np.random.normal(loc=0., scale=objective.scale, size=_fx.shape[0])[:, None]
    _y = _fx + _noise   # same shape

    x_file, y_file = name_attr(dict_i)

    save_attr(path_to_data + x_file + '.npy', array=_x)     # str.replace is not in-place modificattion of filename
    save_attr(path_to_data + y_file + '.npy', array=_y)


# print(dict_args)
# print(objective.fmin)
# print(_x.shape)
# print(_noise.shape)
# print(_y.shape)
aaa = 5