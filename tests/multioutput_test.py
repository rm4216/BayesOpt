import gpflow
import numpy as np

import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf

import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.models.gplvm_models import NN_SVGP, NN


# D = int(2)
# output = int(6)     # number of outputs
# rank = int(6)       # number of latent GPs
# M = int(100)         # number of inducing points
#
# Xnn = np.load('/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/multioutput_test/'
#               'Xnn_sample.npy')
# X = np.load('/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/multioutput_test/X_'
#             'sample.npy')
# Y = np.load('/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/tests/multioutput_test/Y_'
#             'sample.npy')

D = int(10)
output = int(101)       # number of outputs
rank = int(20)         # number of latent GPs
M = int(150)            # number of inducing points
np.random.seed(123)
Xnn = np.random.uniform(low=0., high=1., size=[300, 10])
from datasets.tasks.all_tasks import ProductSines10D
objective = ProductSines10D()
np.random.seed(23)
X = np.random.uniform(low=0., high=1., size=[300, 100])
Y = objective.f(X, fulldim=False, noisy=True)


def _kern():
    return gpflow.kernels.Matern32(input_dim=D, ARD=True, lengthscales=np.ones(shape=[D]) * 0.2)

np.random.seed(123)
with gpflow.defer_build():

    W = np.random.normal(loc=0., scale=1., size=[output, rank])
    kernels = mk.SeparateMixedMok([_kern() for _ in range(rank)], W)
    # kernels = mk.SharedIndependentMok(gpflow.kernels.Matern32(input_dim=D, ARD=True, lengthscales=np.ones(shape=[D]) * 0.2), output)

    feature_list = [gpflow.features.InducingPoints(Xnn[:M, :]) for r in range(rank)]
    feature = mf.MixedKernelSeparateMof(feature_list)
    # feature = mf.MixedKernelSharedMof(gpflow.features.InducingPoints(Xnn[:M,...].copy()))
    # feature = mf.SharedIndependentMof(gpflow.features.InducingPoints(Xnn[:M,...].copy()))

    q_mu = np.zeros((M, rank))
    q_sqrt = np.repeat(np.eye(M)[None, ...], rank, axis=0) * 1.0

    likelihood = gpflow.likelihoods.Gaussian()
    likelihood.variance = 0.01

    XY = np.concatenate([X, Y], axis=1)

    nn = NN(dims=[X.shape[1], 105, D], N=0, proj_dim=0, name=None)
    # model = NN_SVGP(X=X, Y=XY, kern=kernels, likelihood=likelihood, nn=nn,
    #                 feat=feature, q_mu=q_mu, q_sqrt=q_sqrt,
    #                 whiten=False,
    #                 minibatch_size=None,
    #                 num_data=X.shape[0])
    model = gpflow.models.SVGP(X=Xnn, Y=XY, kern=kernels, likelihood=likelihood,
                               feat=feature, q_mu=q_mu, q_sqrt=q_sqrt,
                               whiten=False,
                               minibatch_size=None,
                               num_data=Xnn.shape[0])
    model.compile()

print('Log likelihood before opt: ', model.compute_log_likelihood())
model.as_pandas_table()
gpflow.train.ScipyOptimizer().minimize(model)
model.as_pandas_table()
print('Log likelihood after opt: ', model.compute_log_likelihood())

# mean_X, var_X = model.predict_f(Xnn)
# Xnn = model.nn.np_forward(X)
mean_X, var_X = model.predict_f(Xnn)

import matplotlib.pyplot as plt
f, axs = plt.subplots(X.shape[1] + 1, 1, figsize=(8, 4*X.shape[1]), sharex=True)
x_num = np.arange(Xnn.shape[0])
# XY = X
labels_y = ['x0', 'x1', 'x2', 'x3', 'x4', 'y']
for i, ax in zip(range(X.shape[1] + 1), axs):
    ax.errorbar(x_num, mean_X[:, i], yerr=2*np.sqrt(var_X[:, i]), fmt='o', color='blue', label='mogp x')
    ax.plot(x_num, XY[:, i], '+', color='r', label='true x')
    ax.set_ylabel(labels_y[i])
    ax.set_xlabel('number')
plt.legend()
path = '/home/rm4216/Desktop/ImperialCollege/Python/Github_qgp_bo/BayesOpt/tests/test_manifold/multioutput_test/decoder_test.pdf'
plt.savefig(path, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None)