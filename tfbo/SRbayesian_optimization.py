import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
import argparse
from collections import OrderedDict
import numpy as np
from tfbo.utils.load_save import save_dictionary
from tfbo.utils.name_file import name_synthetic


# dictionary optimization inputs
# seed_number, objective_name, optimizer_name, loss_name, proj_dim, input_dim, maxiter
names = ['seed', 'obj', 'opt', 'loss', 'proj_dim', 'input_dim', 'maxiter', 'part']
defaults = ['0', 'SwissRoll1D', 'SRNNKL_bo', 'Neg_ei', int(1), int(2), int(5), int(10)]
types = [str, str, str, str, int, int, int, int]
helps = ['seed_number', 'objective_name', 'optimizer_name', 'loss_name', 'proj_dim', 'input_dim', 'maxiter', 'proportion_of_SR']

parser = argparse.ArgumentParser(description='Input: seed_number, objective_name, optimizer_name, loss_name, proj_dim,'
                                             ' input_dim, maxiter, proportion_of_SR')
for name_i, default_i, type_i, help_i in zip(names, defaults, types, helps):
    parser.add_argument('--' + name_i, default=default_i, type=type_i, help=help_i)
args = parser.parse_args()
dict_args = vars(args)
print(dict_args)

# check inputs
verify_attr = import_attr('tfbo/utils/check_inputs', attribute='verify_dict_inputs')
verify_attr(dict_args)
string_to_int_attr = import_attr('tfbo/utils/check_inputs', attribute='transform_inputs')
dict_args['seed'] = string_to_int_attr(dict_args['seed'])


# load start
load_start_attr = import_attr('tfbo/utils/load_save', attribute='load_SRinitializations')
xy_list = load_start_attr(dict_args, names)

# load optimizer
path_opt = 'tfbo/optimizers/' + dict_args['opt'] + '_optimizer'
optim_attr = import_attr(path_opt, attribute=dict_args['opt'] + '_optimizer')

# load objective
path_obj = 'datasets/tasks/all_tasks'
task_attr = import_attr(path_obj, attribute=dict_args['obj'])
objective = task_attr()

if dict_args['input_dim'] == dict_args['proj_dim']:
    xy_list_input_dim = [(xy_list[i][0][:, :, objective.relevant_dims], xy_list[i][1]) for i in range(len(xy_list))]
    y_list = []
    for j in range(len(xy_list)):
        y_list.append([objective.f(xy_list_input_dim[j][0][0, i, :], noisy=True, fulldim=True) for i in
                       range(xy_list_input_dim[j][0].shape[1])])
else:
    xy_list_input_dim = xy_list

# maxiters
maxiters = dict_args['maxiter']     # int(100)

def optimizer_run(xy_i, seed_i):
    # initialize new optimizer
    kwargs = {'seed': seed_i, 'dict_args': dict_args}
    optimizer = optim_attr(xy_i, proj_dim=dict_args['proj_dim'], objective=objective, loss=dict_args['loss'], **kwargs)
    # results optimization
    x_out, y_out, hyp_out, lik_out = optimizer.run(maxiters=maxiters)
    return (x_out, y_out, hyp_out, lik_out)
list_outs = list(map(optimizer_run, xy_list_input_dim, dict_args['seed']))


dict_out = OrderedDict(
    [
        ('Xepisodes', list_outs[0][0][None]),
        ('Yepisodes', list_outs[0][1]),
        ('hyp_episodes', np.stack(list_outs[0][2], axis=0)[None])
    ])
path_dict = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/results/'
filename = name_synthetic(dict_args)
save_dictionary(path_dict + filename + '_part' + str(dict_args['part']) + 'seed_' + str(dict_args['seed'][0]) + '.p', dict_out)