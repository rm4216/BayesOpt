import tensorflow as tf
import numpy as np
import gpflow
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from tfbo.models.gplvm_models import FullMoGP


np.random.seed(123)

input_dim = int(60)
N = int(500)
output_dim = input_dim

X = np.random.uniform(low=0., high=1., size=[N, input_dim])
task_attr = import_attr('datasets/tasks/all_tasks', attribute='ProductSinesLinear10D')
objective = task_attr()
Y = objective.f(X, fulldim=False, noisy=True)

k0 = gpflow.kernels.Matern52(input_dim=input_dim, ARD=True, active_dims=list(range(input_dim)), lengthscales=np.ones(shape=[input_dim])*0.2)
k1 = gpflow.kernels.Coregion(input_dim=1, output_dim=output_dim, rank=output_dim, active_dims=[input_dim])
kernel = k0 * k1
np.random.seed(23)
kernel.kernels[1].W = np.random.randn(output_dim, output_dim)

gpMo = FullMoGP(X=X, Y=Y, kern=kernel, Mo_dim=output_dim)
gpMo.as_pandas_table()
logpdf_qlq = gpMo.compute_log_likelihood()



K_np = kernel.kernels[0].compute_K_symm(X)
B_np = np.matmul(kernel.kernels[1].W.read_value(), np.transpose(kernel.kernels[1].W.read_value())) + \
       np.diag(kernel.kernels[1].kappa.read_value())
# # Compute Kernel and Coregionalization separately
# K, B = gpMo.compute_log_likelihood()
# err_B = np.max(np.abs(B - B_np))
# err_K = np.max(np.abs(K - K_np))


# Efficient matrix vector multiplication complexity O(N*D)
BK_np = np.kron(B_np, K_np)
X_vec = np.reshape(np.transpose(X), newshape=[N * input_dim, 1])
BKx = np.matmul(BK_np, X_vec)

Gk = np.shape(K_np)[0]
X_Gk = np.transpose(np.reshape(X_vec, [input_dim, Gk]))
Z = np.matmul(K_np, X_Gk)
Z = np.transpose(Z)

Z_vec = np.reshape(np.transpose(Z), [N * input_dim, 1])
Gb = np.shape(B_np)[0]
Z_Gb = np.transpose(np.reshape(Z_vec, [N, Gb]))
M = np.matmul(B_np, Z_Gb)
M = np.transpose(M)

x_out = np.reshape(np.transpose(M), [N * input_dim, 1])
err_x = np.max(np.abs(BKx - x_out))

# x_out_tf = gpMo.compute_log_likelihood()
# err_kron = np.max(np.abs(BKx - x_out_tf))




# Matrix eigen-decomposition
lmbda_k, Q_k = np.linalg.eig(K_np)
K_eig = np.matmul(np.matmul(Q_k, np.diag(lmbda_k)), np.transpose(Q_k))
err_Keig = np.max(np.abs(K_eig - K_np))

lmbda_b, Q_b = np.linalg.eig(B_np)
B_eig = np.matmul(np.matmul(Q_b, np.diag(lmbda_b)), np.transpose(Q_b))
err_Beig = np.max(np.abs(B_eig - B_np))

# K_new, B_new, K_old, B_old = gpMo.compute_log_likelihood()
# err_Knew = np.max(np.abs(K_new - K_old))
# err_Bnew = np.max(np.abs(B_new - B_old))


# QbQkX_vec, Inv_vec, alpha = gpMo.compute_log_likelihood()
alpha_np = np.matmul(np.linalg.inv(BK_np + np.eye(N*input_dim) * gpMo.likelihood.variance.read_value()), X_vec)
# err_inverse = np.max(np.abs(alpha_np - alpha))

# logpdf_qlq, logpdf_L = gpMo.compute_log_likelihood()
logpdf_qlq = gpMo.compute_log_likelihood()
# L_mat = gpMo.compute_log_likelihood()
err_lpdf = np.abs(logpdf_qlq - logpdf_L)

a = 5.
