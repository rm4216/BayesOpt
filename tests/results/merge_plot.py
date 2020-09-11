import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..', '..'))
from tfbo.utils.load_save import load_dictionary
from tfbo.utils.import_modules import import_attr
from collections import OrderedDict
from tfbo.utils.name_file import name_synthetic_dict_no_quantile
from tfbo.configurations_plot import generate_ax, plot_results, plot_threshold
import numpy as np
import matplotlib.pyplot as plt


dict_args = OrderedDict(
    [
        ('seed', int(0)),
        ('obj', 'ProductSinesLinear10D'),
        ('opt', 'NN_bo'),
        ('loss', 'Neg_pi'),
        ('proj_dim', int(10)),
        ('input_dim', int(60))
    ])
path_dict = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/results/'
# filename = 'test_MgpOpt'
# dict_input = load_dictionary(path_dict + filename + '.p')
filename = name_synthetic_dict_no_quantile(dict_args)
# dict_input = load_dictionary(path_dict + filename + 'seed_' + str(dict_args['seed']) + '.p')
dict_input = load_dictionary(path_dict + filename + '.p')


task_attr = import_attr('datasets/tasks/all_tasks', attribute=dict_args['obj'])
objective = task_attr()
# f_out = objective.f(np.array([0.50313, 0.18502]), noisy=False, fulldim=True)
# f_out2 = objective.f(objective.minimizer1, noisy=False, fulldim=False)

xiters = dict_input['NNKL_bo']['Xepisodes'].shape[1]

if dict_args['obj'] == 'Branin2D':
    f, ax = generate_ax(title='', xlabel='iterations', ylabel='best f', xlim=[-1, xiters + 1.],
                    ylim=[objective.fmin - 0.5, 25.])
elif dict_args['obj'] == 'Hartmann6D':
    f, ax = generate_ax(title='', xlabel='iterations', ylabel='best f', xlim=[-1, xiters + 1.],
                        ylim=[objective.fmin - 0.5, 0.5])
elif dict_args['obj'] == 'HartmannLinear6D':
    f, ax = generate_ax(title='', xlabel='iterations', ylabel='best f', xlim=[-1, xiters + 1.],
                        ylim=[objective.fmin - 0.5, 0.5])
elif dict_args['obj'] == 'ProductSinesLinear10D':
    f, ax = generate_ax(title='', xlabel='iterations', ylabel='best f', xlim=[-1, xiters + 1.],
                        ylim=[objective.fmin - 0.5, 0.5])
elif dict_args['obj'] == 'MichalewiczLinear10D':
    f, ax = generate_ax(title='', xlabel='iterations', ylabel='best f', xlim=[-1, xiters + 1.],
                    ylim=[objective.fmin - 0.5, 0.])
elif dict_args['obj'] == 'RosenbrockLinear10D':
    f, ax = generate_ax(title='', xlabel='iterations', ylabel='best f', xlim=[-1, xiters + 1.],
                    ylim=[objective.fmin - 0.5, 1000])
elif dict_args['obj'] == 'ShekelLinear4D':
    f, ax = generate_ax(title='', xlabel='iterations', ylabel='best f', xlim=[-1, xiters + 1.],
                    ylim=[objective.fmin - 0.5, 0.])

names  = ['NN_bo']   + ['vae_bo']     + ['NNKL_bo'] + ['NNKL2_bo']+ ['add_bo']  + ['rembo']
labels = ['NN BO']   + ['VAE BO']     + ['KL BO'] + ['KL2 BO'] + ['ADD BO']  + ['REMBO']
colors = ['blue']    + ['darkorange'] + ['red'] + ['yellow'] + ['magenta'] + ['aqua']
linestyles = ['-'] + ['--'] + ['--'] + ['--'] + ['--'] + ['--']
alphas = [0.2] + [0.2] + [0.2] + [0.2] + [0.2] + [0.2]


# names  = ['NN_bo']   + ['add_bo']   # + ['rembo']
# labels = ['NN BO']   + ['ADD BO']   # + ['REMBO']
# colors = ['blue']   + ['magenta']   # + ['aqua']
# linestyles = ['--'] + ['--']
# alphas = [0.2] + [0.2]

xiterations = np.arange(xiters)[:, None]
for name_i, label_i, color_i, line_i, alpha_i in zip(names, labels, colors, linestyles, alphas):
    ax = plot_results(ax, x=xiterations, y=dict_input[name_i]['Yepisodes'][:xiters, :], label=label_i, linestyle=line_i,
                      color=color_i, alpha=alpha_i)
ax = plot_threshold(ax, xiterations, threshold=objective.fmin, label='fmin', linestyle='--', color='k')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)

lgd = ax.legend(loc=3, bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, mode="expand", borderaxespad=0.)
plt.savefig(filename + '.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, bbox_extra_artists=(lgd,))