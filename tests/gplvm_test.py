import gpflow
from gpflow import kernels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tfbo.models.gplvm_models import BC_GPLVM, NN

np.random.seed(123)
gpflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations

X = np.random.uniform(low=0., high=1., size=[100, 5])
kern = gpflow.kernels.RBF(input_dim=5, ARD=True, variance=5., lengthscales=np.array([0.3, 2., 5., 0.6, 0.11]))
KXX = kern.compute_K_symm(X)
L = np.linalg.cholesky(KXX + np.eye(X.shape[0]) * 1e-06)
n = np.random.normal(loc=0., scale=1., size=[X.shape[0], 1])
fX = np.matmul(L, n)
Y = fX + np.random.normal(loc=0., scale=0.01, size=[X.shape[0], 1])

# Model with NN(X) that applies multi-output GP(NN(X)) ->[X, y]
# Trains jointly the multi-output GP and the neural network to map to the original space and the outputs
proj_dim = int(2)

k1 = gpflow.kernels.RBF(input_dim=proj_dim, ARD=True, active_dims=list(range(proj_dim)))
coreg = gpflow.kernels.Coregion(input_dim=1, output_dim=X.shape[1] + 1, rank=1, active_dims=[proj_dim])
coreg.W = np.random.randn(X.shape[1] + 1, 1)
kc = k1 * coreg

nn = NN(dims=[X.shape[1], 8, proj_dim], N=0, proj_dim=0, name=None)    # otherwise re-initialized at each BO iteration

from tfbo.models.gplvm_models import NN_MOGPR
nn_mogp = NN_MOGPR(X=X, Y=Y, kern=kc, nn=nn)
nn_mogp.as_pandas_table()
nn_mogp.read_trainables()
nn_mogp.compute_log_likelihood()

gpflow.train.ScipyOptimizer().minimize(nn_mogp)

nn_mogp.as_pandas_table()
nn_mogp.compute_log_likelihood()
Xnn = nn_mogp.nn.np_forward(X)

# Recover training locations x_{3} and output y_{3} from the multiple output predictions
x_test2D = np.copy(Xnn[3, :])[None]
x_test2D_aug = np.vstack([np.hstack([x_test2D, np.ones(shape=[1, 1]) * i]) for i in range(X.shape[1] + 1)])
x_mean, x_var = nn_mogp.predict_f(x_test2D_aug)
x_train = np.copy(X[3, :])[None]
y_train = np.copy(Y[3, :])[None]
x_mean_test = x_mean[:-1, :]
y_mean_test = x_mean[-1, :]

# f, axs = plt.subplots(X.shape[1] + 1, 1, figsize=(8, 4*X.shape[1]), sharex=True)
# x_num = np.arange(Xnn.shape[0])
# XY = np.hstack([X, Y])
# labels_y = ['x0', 'x1', 'x2', 'x3', 'x4', 'y']
# for i, ax in zip(range(X.shape[1]+1), axs):
#     x_test_i = np.hstack([Xnn, np.ones(shape=[Xnn.shape[0], 1]) * i])
#     x_mean_i, x_var_i = nn_mogp.predict_f(x_test_i)
#     ax.plot(x_num, XY[:, i], '+', color='r', label='true x')
#     ax.errorbar(x_num, x_mean_i, yerr=2*np.sqrt(x_var_i), fmt='o', color='blue', label='mogp x')
#     ax.set_ylabel(labels_y[i])
#     ax.set_xlabel('number')
# plt.legend()
# plt.savefig('multiple_outputs_GP.pdf', dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format='pdf',
#             transparent=False, bbox_inches=None, pad_inches=0.1,
#             frameon=None)

x1 = np.linspace(start=0., stop=1., num=100)[:, None]
X1, X2 = np.meshgrid(x1, x1)
XX = np.column_stack([np.ravel(X1), np.ravel(X2)])
XX_aug = np.hstack([XX, np.ones(shape=[XX.shape[0], 1]) * (X.shape[1])])
mean_xx, var_xx = nn_mogp.predict_f(XX_aug)

from mpl_toolkits.mplot3d import Axes3D
fig_new = plt.figure()
ax3d = fig_new.add_subplot(111, projection='3d')
ax3d.plot_surface(X1, X2, np.reshape(mean_xx, newshape=X1.shape), label='mean')
ax3d.plot_surface(X1, X2, np.reshape(mean_xx, newshape=X1.shape) - 2 * np.reshape(np.sqrt(var_xx), newshape=X1.shape), alpha=0.2, color='blue', label='$2\sigma$')
ax3d.plot_surface(X1, X2, np.reshape(mean_xx, newshape=X1.shape) + 2 * np.reshape(np.sqrt(var_xx), newshape=X1.shape), alpha=0.2, color='blue')
ax3d.scatter(Xnn[:, 0], Xnn[:, 1], Y, c='r', marker='o', label='manifold data')
ax3d.set_xlabel('X1')
ax3d.set_ylabel('X2')
ax3d.set_zlabel('Z')

# ax3d.legend()
plt.savefig('manifold_GP.pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None)