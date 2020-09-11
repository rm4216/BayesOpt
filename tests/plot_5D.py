import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

k=0.4


f, axarr = plt.subplots(2,5, sharex='col', sharey='row')
f2s, axarr2s = plt.subplots(2,5, sharex='col', sharey='row')
fs, axarrs = plt.subplots(2,5, sharex='col', sharey='row')


x = np.linspace(start=0., stop=1., num=100)[:, None]
y = np.linspace(start=0., stop=1., num=100)[:, None]
X,Y = np.meshgrid(x, y)
XY = np.column_stack([np.ravel(X), np.ravel(Y)])
x_column = XY[:, 0].copy()
y_column = XY[:, 1].copy()

start_vec = np.ones(shape=[x_column.shape[0], 5]) * 0.5

pairs_ij = []
Fs = []
Vs = []

for i in range(5-1):
    for j in range(i+1,5):
        print('[' + str(i) + ', ' + str(j) + ']')
        predict_vec = np.copy(start_vec)
        predict_vec[:, i] = x_column
        predict_vec[:, j] = y_column
        fmean = np.ones_like(x_column) * k
        fvar = np.ones_like(x_column) * k * 0.1

        pairs_ij.append((i, j))
        Fs.append(np.reshape(fmean, newshape=[X.shape[0], X.shape[1]]))
        Vs.append(np.reshape(np.sqrt(np.abs(fvar)), newshape=[X.shape[0], X.shape[1]]))

axarray_used = list(np.ravel(axarr))
for ax_i, ax2s_i, axs_i, pair_i, F, V in zip(list(np.ravel(axarr)), list(np.ravel(axarr2s)), list(np.ravel(axarrs)), pairs_ij, Fs, Vs):
    i = pair_i[0]
    j = pair_i[1]
    ax_i.pcolor(X, Y, F, cmap=cm.RdBu)
    ax_i.set_title('Dims [' + str(i) +  ', ' + str(j) + ']')
    # ax_i.set_xlabel('x' + str(i))
    # ax_i.set_ylabel('x' + str(j))

    ax2s_i.pcolor(X, Y, F + 2. * V, cmap=cm.RdBu)
    ax2s_i.set_title('Dims [' + str(i) +  ', ' + str(j) + ']')
    # ax2s_i.set_xlabel('x' + str(i))
    # ax2s_i.set_ylabel('x' + str(j))

    axs_i.pcolor(X, Y, F + 2. * V, cmap=cm.RdBu)
    axs_i.set_title('Dims [' + str(i) +  ', ' + str(j) + ']')
    # axs_i.set_xlabel('x' + str(i))
    # axs_i.set_ylabel('x' + str(j))


plt.show()
aaa = 5.
# p = ax.pcolor(X, Y, fmean, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
# cb = fig.colorbar(p)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

f, axarr = plt.subplots(2, 5, sharex='col', sharey='row')
f2s, axarr2s = plt.subplots(2, 5, sharex='col', sharey='row')
fs, axarrs = plt.subplots(2, 5, sharex='col', sharey='row')

x = np.linspace(start=0., stop=1., num=100)[:, None]
y = np.linspace(start=0., stop=1., num=100)[:, None]
X, Y = np.meshgrid(x, y)
XY = np.column_stack([np.ravel(X), np.ravel(Y)])
x_column = XY[:, 0].copy()
y_column = XY[:, 1].copy()

start_vec = np.tile(x_proj_tp1, [x_column.shape[0], 1])  # ones(shape=[x_column.shape[0], 5]) * 0.5

pairs_ij = []
Fs = []
Vs = []

for i in range(5 - 1):
    for j in range(i + 1, 5):
        print('[' + str(i) + ', ' + str(j) + ']')
        predict_vec = np.copy(start_vec)
        predict_vec[:, i] = x_column
        predict_vec[:, j] = y_column
        fmean, fvar = gp_nnjoint.predict_f(predict_vec)

        pairs_ij.append((i, j))
        Fs.append(np.reshape(fmean, newshape=[X.shape[0], X.shape[1]]))
        Vs.append(np.reshape(np.sqrt(np.abs(fvar)), newshape=[X.shape[0], X.shape[1]]))

axarray_used = list(np.ravel(axarr))
for ax_i, ax2s_i, axs_i, pair_i, F, V in zip(list(np.ravel(axarr)), list(np.ravel(axarr2s)), list(np.ravel(axarrs)),
                                             pairs_ij, Fs, Vs):
    i = pair_i[0]
    j = pair_i[1]
    ax_i.pcolor(X, Y, F, cmap=cm.RdBu)
    ax_i.set_title('Dims [' + str(i) + ', ' + str(j) + ']')
    ax_i.plot(x_proj_tp1[0, i], x_proj_tp1[0, j], '*')
    # ax_i.set_xlabel('x' + str(i))
    # ax_i.set_ylabel('x' + str(j))

    ax2s_i.pcolor(X, Y, F + 2. * V, cmap=cm.RdBu)
    ax2s_i.set_title('Dims [' + str(i) + ', ' + str(j) + ']')
    # ax2s_i.set_xlabel('x' + str(i))
    # ax2s_i.set_ylabel('x' + str(j))

    axs_i.pcolor(X, Y, F + 2. * V, cmap=cm.RdBu)
    axs_i.set_title('Dims [' + str(i) + ', ' + str(j) + ']')
    # axs_i.set_xlabel('x' + str(i))
    # axs_i.set_ylabel('x' + str(j))











import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

f, axarr = plt.subplots(2, 5, sharex='col', sharey='row')

x = np.linspace(start=0., stop=1., num=100)[:, None]
y = np.linspace(start=0., stop=1., num=100)[:, None]
X, Y = np.meshgrid(x, y)
XY = np.column_stack([np.ravel(X), np.ravel(Y)])
x_column = XY[:, 0].copy()
y_column = XY[:, 1].copy()

start_vec = np.tile(x_proj_tp1, [x_column.shape[0], 1])  # ones(shape=[x_column.shape[0], 5]) * 0.5

pairs_ij = []
Fs = []
Vs = []

for i in range(5 - 1):
    for j in range(i + 1, 5):
        print('[' + str(i) + ', ' + str(j) + ']')
        predict_vec = np.copy(start_vec)
        predict_vec[:, i] = x_column
        predict_vec[:, j] = y_column
        fmean, fvar, fg = acquisition(predict_vec)

        pairs_ij.append((i, j))
        Fs.append(np.reshape(fmean, newshape=[X.shape[0], X.shape[1]]))

axarray_used = list(np.ravel(axarr))
for ax_i, pair_i, F in zip(list(np.ravel(axarr)),
                                             pairs_ij, Fs):
    i = pair_i[0]
    j = pair_i[1]
    ax_i.pcolor(X, Y, F, cmap=cm.RdBu)
    ax_i.set_title('Dims [' + str(i) + ', ' + str(j) + ']')
    ax_i.plot(x_proj_tp1[0, i], x_proj_tp1[0, j], '*', color='white')
    # ax_i.set_xlabel('x' + str(i))
    # ax_i.set_ylabel('x' + str(j))
plt.savefig('acquisitions' + '.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

f, axarr = plt.subplots(2, 5, sharex='col', sharey='row')

x = np.linspace(start=0., stop=1., num=100)[:, None]
y = np.linspace(start=0., stop=1., num=100)[:, None]
X, Y = np.meshgrid(x, y)
XY = np.column_stack([np.ravel(X), np.ravel(Y)])
x_column = XY[:, 0].copy()
y_column = XY[:, 1].copy()

start_vec = np.tile(x_proj_tp1, [x_column.shape[0], 1])  # ones(shape=[x_column.shape[0], 5]) * 0.5

pairs_ij = []
Fs = []
Vs = []

for i in range(5 - 1):
    for j in range(i + 1, 5):
        print('[' + str(i) + ', ' + str(j) + ']')
        predict_vec = np.copy(start_vec)
        predict_vec[:, i] = x_column
        predict_vec[:, j] = y_column
        fmean, fvar = gp_nnjoint.predict_f(predict_vec)

        pairs_ij.append((i, j))
        Fs.append(np.reshape(fmean, newshape=[X.shape[0], X.shape[1]]))

axarray_used = list(np.ravel(axarr))
for ax_i, pair_i, F in zip(list(np.ravel(axarr)),
                                             pairs_ij, Fs):
    i = pair_i[0]
    j = pair_i[1]
    ax_i.pcolor(X, Y, F, cmap=cm.RdBu)
    ax_i.set_title('Dims [' + str(i) + ', ' + str(j) + ']')
    ax_i.plot(x_proj_tp1[0, i], x_proj_tp1[0, j], '*', color='white')
    # ax_i.set_xlabel('x' + str(i))
    # ax_i.set_ylabel('x' + str(j))
plt.savefig('GP_means' + '.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
