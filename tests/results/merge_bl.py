import argparse
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..', '..'))
from tfbo.utils.import_modules import import_attr
from tfbo.utils.store_outputs import initialize_dictionary
from copy import deepcopy


names = ['seed', 'obj', 'opt', 'loss', 'proj_dim', 'input_dim', 'maxiter']
defaults = ['0', 'ProductSinesLinear10D', 'NN_bo', 'Neg_pi', int(10), int(60), int(50)]    # list ints, name_objective, name_optimizer, name loss
types = [str, str, str, str, int, int, int, float]
parser = argparse.ArgumentParser(description='Input: list numbers, name_objective, name_optimizer, name loss, proj_dim, input_dim, maxiters')
for name_i, default_i, type_i in zip(names, defaults, types):
    parser.add_argument('--' + name_i, default=default_i, type=type_i)
args = parser.parse_args()
dict_args = vars(args)
print(dict_args)
string_to_int_attr = import_attr('tfbo/utils/check_inputs', attribute='transform_inputs')
dict_args['seed'] = string_to_int_attr(dict_args['seed'])

import_name_attr = import_attr('tfbo/utils/name_file', attribute='name_synthetic')
import_name_save_attr = import_attr('tfbo/utils/name_file', attribute='name_synthetic_dict_no_quantile')
import_dict_attr = import_attr('tfbo/utils/load_save', attribute='load_dictionary')
save_dict_attr = import_attr('tfbo/utils/load_save', attribute='save_dictionary')

path = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/results/'

acqs = ['Neg_ei', 'lcb', 'Neg_pi']
# opts = ['NN_bo', 'add_bo']
opts = ['NN_bo', 'vae_bo', 'NNKL_bo', 'NNKL2_bo', 'add_bo', 'rembo']
for acq_i in acqs:

    dict_args['loss'] = acq_i

    dict_out = initialize_dictionary()

    for opt_i in opts:
        dict_args['opt'] = opt_i
        name_dict = import_name_attr(dict_args)
        dict_input = import_dict_attr(path + name_dict + '.p')
        dict_out[opt_i] = deepcopy(dict_input)

    name_save = import_name_save_attr(dict_args)
    save_dict_attr(path + name_save + '.p', dict_out)