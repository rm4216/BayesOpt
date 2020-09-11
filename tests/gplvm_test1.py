import gpflow
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tfbo.models.gplvm_models import BC_GPLVM, NN, MOGPR, Linear_NN

np.random.seed(123)
gpflow.settings.numerics.quadrature = 'error'  # throw error if quadrature is used for kernel expectations

X = np.random.uniform(low=0., high=1., size=[100, 5])
kern = gpflow.kernels.RBF(input_dim=5, ARD=True, variance=5., lengthscales=np.array([0.3, 2., 5., 0.6, 0.11]))
KXX = kern.compute_K_symm(X)
L = np.linalg.cholesky(KXX + np.eye(X.shape[0]) * 1e-06)
n = np.random.normal(loc=0., scale=1., size=[X.shape[0], 1])
fX = np.matmul(L, n)
Y = fX + np.random.normal(loc=0., scale=0.01, size=[X.shape[0], 1])

Xnorm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Ynorm = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)


# Bayesian GP-LVM and NN transforming the variational parameters of the mean only
proj_dim = int(2)
# nn = NN(dims=[X.shape[1], 8, proj_dim], N=X.shape[0], proj_dim=proj_dim)
nn = Linear_NN(dims=[X.shape[1], 8, proj_dim], N=X.shape[0], proj_dim=proj_dim)

k1 = gpflow.kernels.Matern32(input_dim=proj_dim, ARD=True, active_dims=list(range(proj_dim)))
coreg = gpflow.kernels.Coregion(input_dim=1, output_dim=X.shape[1], rank=X.shape[1], active_dims=[proj_dim])
coreg.W = np.random.randn(X.shape[1], X.shape[1])
kc = k1 * coreg
gp_ae = MOGPR(X=X, Y=Y, kern=kc, nn=nn)
# gp_ae.kern.kernels[0].variance = 12.
gp_ae.likelihood.variance = 0.01

gp_ae.as_pandas_table()
lik_ae0 = gp_ae.compute_log_likelihood()
gpflow.train.ScipyOptimizer().minimize(gp_ae)
Xnn = gp_ae.nn.np_forward(X)
# f, axs = plt.subplots(X.shape[1], 1, figsize=(8, 4*X.shape[1]), sharex=True)
# x_num = np.arange(Xnn.shape[0])
# XY = X
# labels_y = ['x0', 'x1', 'x2', 'x3', 'x4']
# for i, ax in zip(range(X.shape[1]), axs):
#     x_test_i = np.hstack([Xnn, np.ones(shape=[Xnn.shape[0], 1]) * i])
#     x_mean_i, x_var_i = gp_ae.predict_f(x_test_i)
#     ax.errorbar(x_num, x_mean_i, yerr=2*np.sqrt(x_var_i), fmt='o', color='blue', label='mogp x')
#     ax.plot(x_num, XY[:, i], '+', color='r', label='true x')
#     ax.set_ylabel(labels_y[i])
#     ax.set_xlabel('number')
# plt.legend()
# plt.savefig('ae_GP.pdf', dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format='pdf',
#             transparent=False, bbox_inches=None, pad_inches=0.1,
#             frameon=None)
gp_ae.as_pandas_table()
lik_ae1 = gp_ae.compute_log_likelihood()


kernel = gpflow.kernels.RBF(input_dim=proj_dim, ARD=True, active_dims=list(range(proj_dim)), lengthscales=np.ones(shape=[proj_dim])*0.2)
# kernel = gpflow.kernels.RBF(input_dim=5, ARD=True, active_dims=list(range(5)))
# Xnn_norm = ( Xnn - np.mean(Xnn, axis=0) ) / np.std(Xnn, axis=0)
b_gplvm = gpflow.models.BayesianGPLVM(X_mean=Xnn, X_var=0.1*np.ones_like(Xnn), Y=Ynorm, kern=kernel, M=100)
b_gplvm.likelihood.variance = 0.01
print(b_gplvm.as_pandas_table())
lik_b0 = b_gplvm.compute_log_likelihood()
gpflow.train.ScipyOptimizer().minimize(b_gplvm)
print(b_gplvm.as_pandas_table())
lik_b1 = b_gplvm.compute_log_likelihood()







# bc_gplvm = BC_GPLVM(X=X, proj_dim=proj_dim, Y=Y, kern=kernel, M=20, nn=nn)
# bc_gplvm.nn.W_0.trainable = False
# bc_gplvm.nn.b_0.trainable = False
# bc_gplvm.nn.W_1.trainable = False
# bc_gplvm.nn.b_1.trainable = False
# bc_gplvm.likelihood.variance = 0.0001
# bc_gplvm.as_pandas_table()
# lik_bc0 = bc_gplvm.compute_log_likelihood()
# gpflow.train.ScipyOptimizer().minimize(bc_gplvm)
# bc_gplvm.as_pandas_table()
# lik_bc1 = bc_gplvm.compute_log_likelihood()
