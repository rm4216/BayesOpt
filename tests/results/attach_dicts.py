import argparse
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..', '..'))
from tfbo.utils.import_modules import import_attr


names = ['seed', 'obj', 'opt', 'loss', 'proj_dim', 'input_dim', 'maxiter']
defaults = ['0', 'RosenbrockLinear10D', 'NNKL_bo', 'Neg_pi', int(10), int(60), int(50)]    # list ints, name_objective, name_optimizer, name loss
types = [str, str, str, str, int, int, int, float]
parser = argparse.ArgumentParser(description='Input: list numbers, name_objective, name_optimizer, name loss, proj_dim, input_dim, maxiters')
for name_i, default_i, type_i in zip(names, defaults, types):
    parser.add_argument('--' + name_i, default=default_i, type=type_i)
args = parser.parse_args()
dict_args = vars(args)
print(dict_args)
string_to_int_attr = import_attr('tfbo/utils/check_inputs', attribute='transform_inputs')
dict_args['seed'] = string_to_int_attr(dict_args['seed'])

import_dict_attr = import_attr('tfbo/utils/load_save', attribute='load_dictionary')
import_name_attr = import_attr('tfbo/utils/name_file', attribute='name_synthetic')
attach_dicts_attr = import_attr('tfbo/utils/store_outputs', attribute='attach_subdictionaries')

path = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/results/'
name_dict_a = import_name_attr(dict_args)
dict_a = import_dict_attr(path + name_dict_a + 'seed_0' + '.p')
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:    # range(19):     # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
    dict_b = import_dict_attr(path + name_dict_a + 'seed_' + str(i+1) + '.p')   # change this line for future attach files
    dict_out = attach_dicts_attr(dict_a, dict_b)    # does dict_a change outside?

save_dict_attr = import_attr('tfbo/utils/load_save', attribute='save_dictionary')
save_dict_attr(path + name_dict_a + '.p', dict_out)
aa = 5.