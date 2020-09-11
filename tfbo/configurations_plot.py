import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def generate_ax(title, xlabel, ylabel, xlim, ylim):
    # matplotlib.rc('axes', labelsize=15)
    # matplotlib.rc('xtick', labelsize=12)
    # matplotlib.rc('ytick', labelsize=12)
    # matplotlib.rc('legend', fontsize=15)
    fig_gen, ax_gen = plt.subplots(1, 1, figsize=(7.,7.), sharex=True)
    ax_gen.set_xlabel(xlabel)
    ax_gen.set_ylabel(ylabel)
    ax_gen.set_xlim(xlim)
    ax_gen.set_ylim(ylim)
    ax_gen.set_title(title)
    return fig_gen, ax_gen

def plot_results(ax_gen, x, y, label, linestyle, color, alpha):
    ybest = np.minimum.accumulate(y, axis=0)
    ymean = np.mean(ybest, axis=1, keepdims=True)
    ystddev = np.std(ybest, axis=1, keepdims=True)
    ax_gen.plot(x, ymean, label=label, linestyle=linestyle, color=color)
    ax_gen.fill_between(x[:, 0], ymean[:, 0]-2*ystddev[:, 0]/np.sqrt(5.), ymean[:, 0]+2*ystddev[:, 0]/np.sqrt(5.),
                        alpha=alpha, color=color)   # , label=label+' 2SE')
    return ax_gen

def plot_regret(ax_gen, x, y, label, linestyle, color, alpha, objective, Xopt):
    N_experiments = y.shape[1]
    fs = np.concatenate([objective.f(X_i, noisy=False, fulldim=False) for X_i in Xopt], axis=1)
    fs_best = np.minimum.accumulate(fs, axis=0)

    regrets = fs_best - objective.fmin
    log_regrets = np.log10(np.abs(regrets))
    # if np.any(regrets == 0.):
    #     mat = np.where(regrets == 0., np.ones_like(regrets) * np.max(regrets), regrets)
    #     regrets_clip = np.where(regrets == 0., np.ones_like(regrets) * np.min(mat), regrets)
    #     log_regrets = np.log10(np.abs(regrets_clip))
    # else:
    #     log_regrets = np.log10(np.abs(regrets))

    # ybest = np.minimum.accumulate(y, axis=0)
    lr_mean = np.mean(log_regrets, axis=1, keepdims=True)
    lr_stddev = np.std(log_regrets, axis=1, keepdims=True)
    ax_gen.plot(x, lr_mean, label=label, linestyle=linestyle, color=color)
    ax_gen.fill_between(x[:, 0], lr_mean[:, 0]-lr_stddev[:, 0]/np.sqrt(N_experiments), lr_mean[:, 0]+lr_stddev[:, 0]/np.sqrt(N_experiments),    # only one standard error (i.e. divide by sqrt(N_experiments))
                        alpha=alpha, color=color)   # , label=label+' 2SE')
    return ax_gen

# import numpy_indexed as npi
def plot_regret_true(ax_gen, x, y, label, linestyle, color, alpha, objective, Xopt):
    fs = np.concatenate([objective.f(X_i, noisy=False, fulldim=False) for X_i in Xopt], axis=1)
    fs_best = np.minimum.accumulate(fs, axis=0)

    regrets = fs_best - objective.fmin
    log_regrets = np.log10(np.abs(regrets))

    # ybest = np.minimum.accumulate(y, axis=0)
    lr_mean = np.mean(log_regrets, axis=1, keepdims=True)
    lr_stddev = np.std(log_regrets, axis=1, keepdims=True)
    ax_gen.plot(x, lr_mean, label=label, linestyle=linestyle, color=color)
    ax_gen.fill_between(x[:, 0], lr_mean[:, 0]-2*lr_stddev[:, 0]/np.sqrt(5.), lr_mean[:, 0]+2*lr_stddev[:, 0]/np.sqrt(5.),
                        alpha=alpha, color=color)   # , label=label+' 2SE')
    return ax_gen

def plot_threshold(ax_gen, x, threshold, label, linestyle, color):
    ax_gen.plot(x, threshold*np.ones_like(x), label=label, linestyle=linestyle, color=color)
    return ax_gen

def bo_panel_hyps(hyps_result, x_range):
    '''
    :param hyps_result: list of iteration x hyps-values arrays, len=num_episodes
    :param x_range:
    :return: panel showing evolution over iterations of each hyper-parameter
    '''
    all_hyps_vec = np.array(hyps_result)
    d = all_hyps_vec.shape[2] - 2 - 1
    all_hyps_vec[:, :, 0:d+2] = np.exp(all_hyps_vec[:, :, 0:d+2])
    means_hyps = np.mean(all_hyps_vec, axis=0)      # average over episodes
    stddevs_hyps = np.std(all_hyps_vec, axis=0)     # stddev over episodes
    rows = int(np.ceil(np.sqrt(d + 3)));     columns = int(np.ceil(np.sqrt(d + 3)))
    f, axarray = plt.subplots(rows, columns, figsize=(15, 12))
    min_lenscale = np.maximum(0., all_hyps_vec[:, :, 0:d].min())
    max_lenscale = np.minimum(5., all_hyps_vec[:, :, 0:d].max())
    min_amplitude = np.maximum(0., all_hyps_vec[:, :, d].min())
    max_amplitude = np.minimum(15., all_hyps_vec[:, :, d].max())
    min_noise_var = np.maximum(0., all_hyps_vec[:, :, d+1].min())
    max_noise_var = np.minimum(5., all_hyps_vec[:, :, d+1].max())
    min_mean = np.maximum(-100., np.min(all_hyps_vec[:, :, d+2]))
    max_mean = np.minimum(100., np.max(all_hyps_vec[:, :, d+2]))
    all_bounds = [(min_lenscale, max_lenscale)]*d + [(min_amplitude, max_amplitude)] + \
                 [(min_noise_var, max_noise_var)] + [(min_mean, max_mean)]
    names = ['len_'+str(i+1) for i in range(d)] + ['ampl'] + ['noise_var'] + ['mean']
    axarray_used = np.ravel(axarray)[0:d+3]
    for ax_i, mean_i, stddevs_i, bounds_i, names_i in zip(list(axarray_used), list(means_hyps.transpose()),
                                                          list(stddevs_hyps.transpose()), all_bounds, names):
        ax_i.plot(x_range, mean_i)
        ax_i.fill_between(x_range[:, 0], mean_i-2*stddevs_i/np.sqrt(5.), mean_i+2*stddevs_i/np.sqrt(5.),
                          alpha=0.2, color='blue')
        ax_i.set_ylim(bounds_i)
        ax_i.set_ylabel(names_i)
        ax_i.set_xlabel('iterations')
    plt.title('BO panel hyps')   # eaxarray[0][0].set_title('BO panel hyps')
    f.subplots_adjust(hspace=0)
    plt.tight_layout()
    return f, axarray

def gpflow_panel_hyps(hyps_result, x_range, d):
    '''
    :param hyps_result: list of iteration x hyps-values arrays, len=num_episodes
    :param x_range:
    :return: panel showing evolution over iterations of each hyper-parameter
    '''
    all_hyps_vec = np.array(hyps_result)
    # d = all_hyps_vec.shape[2] - 2 - 1
    # all_hyps_vec = np.copy(all_hyps_vec[:, :, [4, 14]])
    means_hyps = np.mean(all_hyps_vec, axis=0)      # average over episodes
    stddevs_hyps = np.std(all_hyps_vec, axis=0)     # stddev over episodes
    rows = int(np.ceil(np.sqrt(d)));     columns = int(np.floor(np.sqrt(d)))
    f, axarray = plt.subplots(rows, columns, figsize=(15, 12))
    min_lenscale = np.maximum(0., all_hyps_vec[:, :, 0:d].min())
    max_lenscale = np.minimum(5., all_hyps_vec[:, :, 0:d].max())
    # min_amplitude = np.maximum(0., all_hyps_vec[:, :, d].min())
    # max_amplitude = np.minimum(15., all_hyps_vec[:, :, d].max())
    # min_noise_var = np.maximum(0., all_hyps_vec[:, :, d+1].min())
    # max_noise_var = np.minimum(5., all_hyps_vec[:, :, d+1].max())
    # min_mean = np.maximum(-100., np.min(all_hyps_vec[:, :, d+2]))
    # max_mean = np.minimum(100., np.max(all_hyps_vec[:, :, d+2]))
    all_bounds = [(min_lenscale, max_lenscale)]*d   #  + [(min_amplitude, max_amplitude)] + [(min_noise_var, max_noise_var)] + [(min_mean, max_mean)]
    names = ['len_'+str(i+1) for i in range(d)]     #  + ['ampl'] + ['noise_var'] + ['mean']
    axarray_used = np.ravel(axarray)[0:d]
    for ax_i, mean_i, stddevs_i, bounds_i, names_i in zip(list(axarray_used), list(means_hyps.transpose()),
                                                          list(stddevs_hyps.transpose()), all_bounds, names):
        ax_i.plot(x_range, mean_i)
        ax_i.fill_between(x_range[:, 0], mean_i-2*stddevs_i/np.sqrt(5.), mean_i+2*stddevs_i/np.sqrt(5.),
                          alpha=0.2, color='blue')
        ax_i.set_ylim(bounds_i)
        ax_i.set_ylabel(names_i)
        ax_i.set_xlabel('iterations')
    plt.title('BO panel hyps')   # eaxarray[0][0].set_title('BO panel hyps')
    f.subplots_adjust(hspace=0)
    plt.tight_layout()
    return f, axarray

def bo_panel_xs(xs_result, x_range):
    all_xs = np.array(xs_result)
    means_xs = np.mean(all_xs, axis=0)      # shape: [num_episodes, iterations, num_dims]
    stddevs_xs = np.std(all_xs, axis=0)
    f_xs, axarray_xs = plt.subplots(all_xs.shape[2], sharex=True, figsize=(15, 12))
    names = ['x'+str(i) for i in range(all_xs.shape[2])]
    for ax_i, mean_i, stddevs_i, names_i in zip(list(axarray_xs), list(means_xs.transpose()),
                                                list(stddevs_xs.transpose()), names):
        ax_i.plot(x_range, mean_i)
        ax_i.fill_between(x_range[:, 0], mean_i-2*stddevs_i/np.sqrt(5.), mean_i+2*stddevs_i/np.sqrt(5.),
                          alpha=0.2, color='green')
        ax_i.set_ylim([0, 1])
        ax_i.set_ylabel(names_i)
        ax_i.set_xlabel('iterations')
    axarray_xs[0].set_title('BO panel, x-coordinates')
    f_xs.subplots_adjust(hspace=0)
    return f_xs, axarray_xs

def bo_panel_y(ys_result, x_range):
    means_ys = np.mean(ys_result, axis=1)
    stddevs_ys = np.std(ys_result, axis=1)
    fig = plt.figure();     ax_ys = fig.gca()
    plt.plot(x_range, means_ys)
    plt.fill_between(x_range[:, 0], means_ys-2*stddevs_ys/np.sqrt(5.), means_ys+2*stddevs_ys/np.sqrt(5.),
                     alpha=0.2, color='red')
    ax_ys.set_ylim([ys_result.min(), ys_result.max()])
    ax_ys.set_ylabel('ys results')
    ax_ys.set_xlabel('iterations')
    ax_ys.set_title('BO panel ys result')
    return fig, ax_ys

# def plot_synthetic(ax_gen, _Xtest, _fepisodes, c, label, linestyle, alpha):
#     _fbest = np.minimum.accumulate(_fepisodes, axis=0)
#     _fmean = np.mean(_fbest, axis=1)[:, None]
#     _fstd = np.std(_fbest, axis=1)[:, None]
#     ax_gen.plot(_Xtest, _fmean, color=c, label=label, linestyle=linestyle)
#     ax_gen.fill_between(_Xtest[:, 0], _fmean[:, 0]-2*_fstd[:, 0]/np.sqrt(5.), _fmean[:, 0]+2*_fstd[:, 0]/np.sqrt(5.),
#                         color=c, alpha=alpha, label=label+'2SE')
#     return ax_gen