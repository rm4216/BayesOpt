import gpflow
import numpy as np
import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.components.initializations import initialize_acquisition
from tfbo.components.initializations import initialize_m_models
from tfbo.models.gplvm_models import NN, Stable_GPR, Ort_NN
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


class BLRNNKL_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.initialize_X()  # check correct shapes
        self.identity = False
        self.Mo_dim = self.data_x.shape[1]
        self.Xnn = None
        self.latent_bound = []
        self.L = 1.


    def initialize_modelM(self):


        nn = Ort_NN(dims=[self.data_x.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
                    name=None)

        k_list, gp_nnjoint = initialize_m_models(x=np.copy(self.Xnorm), y=np.copy(self.Ynorm),
        # k_list, gp_nnjoint = initialize_m_models(x=np.copy(self.X_inf), y=np.copy(self.Ynorm),
                                                 input_dim=self.proj_dim,
                                                 model='BLR',
                                                 kernel='Matern52',
                                                 ARD=True,
                                                 nn=nn)     # last kernel is the Manifold GP kernel

        gp_nnjoint.likelihood.variance = 1e-06  # 0.001

        return k_list, gp_nnjoint, nn


    def generate_x(self, x_proj, gp_joint):

        try:
            mean_new, post_S, V = gp_joint.predict_x(x_proj)
        except:
            jitter = 1e-03
            noise_variance = 1e-03

            uZ = np.copy(gp_joint.get_uZ())
            y = np.transpose(uZ - np.tile(np.transpose(np.zeros(shape=[np.shape(self.Xnn)[0]])), [gp_joint.p, 1]))    # assuming zero mean function
            Kmn = gp_joint.kern.kernels[0].compute_K(np.copy(self.Xnn), np.copy(x_proj))
            Kmm_sigma = gp_joint.kern.kernels[0].compute_K_symm(np.copy(self.Xnn)) + np.eye(np.shape(self.Xnn)[0]) * 1e-06
            Knn = gp_joint.kern.kernels[0].compute_K_symm(np.copy(x_proj))
            uZsT_mean, uZsT_cov = gp_joint.my_base_conditional(Kmn, Kmm_sigma, Knn, y)  # N x P, N x P or P x N x N
            Lpost = np.linalg.cholesky(np.copy(uZsT_cov[0, :, :]) + np.eye(np.shape(uZsT_cov)[-1]) * 1e-06)
            uZs = np.transpose(uZsT_mean + np.matmul(Lpost, np.copy(gp_joint.standard_test.read_value())))

            gZ = np.transpose(np.copy(gp_joint.X.read_value()))  # D x N
            Kprior = (np.copy(gp_joint.alpha.value.value) * np.matmul(uZ, uZ.transpose()))
            Sxx = (1. / noise_variance) * np.matmul(uZ, uZ.transpose()) + Kprior
            LSxx = np.linalg.cholesky(Sxx + np.eye(np.shape(Sxx)[0]) * jitter)
            LSxx_inv = np.linalg.solve(LSxx, np.eye(np.shape(LSxx)[0]))
            Sxx_inv = np.matmul(LSxx_inv.transpose(), LSxx_inv)
            MN = np.matmul((1. / noise_variance) * np.matmul(gZ, uZ.transpose()) + np.matmul(
                np.zeros(shape=[self.Mo_dim, np.shape(gp_joint.standard_train.read_value())[1]]), Kprior), Sxx_inv)
            post_mean = np.matmul(MN, uZs)  # D x M
            mean_new = np.transpose(post_mean)
        # posterior_sample = gp_joint.sample_x(x_proj)

        x_new = (mean_new * self.X_std) + self.X_mean
        # x_new = np.copy(mean_new)
        return self.sigmoid(x_new)          # check shape


    def initialize_X(self):

        self.Y_mean = np.mean(self.data_y, axis=0, keepdims=True)
        self.Y_std = np.std(self.data_y, axis=0, keepdims=True)

        self.X_inf = self.logit(self.data_x)
        self.X_mean = np.mean(self.X_inf, axis=0, keepdims=True)
        self.X_std = np.std(self.X_inf, axis=0, keepdims=True)

        self.Xnorm = (self.X_inf - self.X_mean) / self.X_std
        self.Ynorm = (self.data_y - self.Y_mean) / self.Y_std   # check all shapes

        if np.isnan(self.Ynorm).any(): raise ValueError('Ynorm contains nan')
        def test_(input_norm):
            xm_test = np.mean(input_norm, axis=0, keepdims=True)
            xst_test = np.std(input_norm, axis=0, keepdims=True) - np.ones([1, input_norm.shape[1]])
            boolm_list = [np.abs(xm_i[0]) <= 1e-02 for xm_i in list(xm_test.transpose())]
            bools_list = [np.abs(xs_i[0]) <= 1e-02 for xs_i in list(xst_test.transpose())]
            assert all(boolm_list) and all(bools_list)
        test_(self.Xnorm)
        test_(self.Ynorm)


    def learnLipschitz(self, gp_joint):
        try:
            rand_ind = np.random.choice(self.Xnn.shape[0], 10, replace=False)
            z_jac_opt = np.copy(self.Xnn[rand_ind, :])
            jac_Dxd_list = []
            for i in range(z_jac_opt.shape[0]):
                jac_list = []
                for j in range(self.input_dim):
                    mu_x, jac_ij = gp_joint.jacobian_BLRx(z_jac_opt[i, :][None], j)     # stochastic gradients, because of the sampling uZ
                    jac_list.append(jac_ij)
                jac_Dxd_i = np.concatenate(jac_list, axis=0)
                jac_Dxd_list.append(jac_Dxd_i)
            jac_NxDxd = np.stack(jac_Dxd_list, axis=0)
            self.L = np.max(np.abs(jac_NxDxd))
        except:
            print('Failure in updating the Lipschitz constant, maintaining the previous value')
        # norms_of_Dxd = np.stack([np.linalg.norm(j_i, ord=None, axis=None, keepdims=False) for j_i in jac_Dxd_list], axis=0)
        # self.L = np.max(norms_of_Dxd)

    def NLconstraint(self, opt_config):

        def KLconstraint(self, x):
            xC = np.reshape(x, [self.num_init, self.proj_dim])          # M x d
            KL = self.KLdist(xC, self.Xnn)                              # M x N
            ind_minimizers = np.argmin(KL, axis=1)                      # (M,)

            # x_mean_fast = np.copy(self.X_inf[ind_minimizers, :])
            x_mean_fast = np.copy(self.Xnorm[ind_minimizers, :])

            max_mean_components = np.amax(np.abs(x_mean_fast), axis=1)[:, None]  # M x 1
            rterm = max_mean_components ** 2. / (2 * self.L**2)                  # M x 1
            lterm = np.min(KL, axis=1)[:, None]                             # M x 1
            constraint = lterm - rterm
            return constraint[:, 0]

        f_constraint = lambda x: KLconstraint(self=self, x=x)
        opt_config['constraints'] = NonlinearConstraint(f_constraint, lb=-np.inf, ub=0, jac='2-point')
        return opt_config

    def KLdist(self, xC, X2):
        xCs = np.sum(np.square(xC), axis=1, keepdims=True)          # M x 1
        Xs = np.sum(np.square(X2), axis=1, keepdims=True)           # N x 1
        xXT = np.matmul(xC, np.transpose(X2))                       # M x N
        KL = (xCs + np.transpose(Xs) - 2. * xXT) / 2.               # M x N
        return KL


    def run(self, maxiters=20):
        # opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')
        opt_config = import_attr('tfbo/configurations', attribute='KLacquisition_opt')
        for j in range(maxiters):
            print('iteration: ', j)

            self.reset_graph()

            # initialize model
            k_list, gp_nnjoint, nn = self.initialize_modelM()
            gp_nnjoint.kern.kernels[0].variance = 1.

            try:
                gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
            except:
                try:
                    gp_nnjoint.jitter = 1e-03
                    gp_nnjoint.noise_variance = 1e-03
                    gp_nnjoint.kern.kernels[1].lengthscales = np.ones(shape=[self.proj_dim])
                    gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
                except:
                    print('Failure in optimization of hyper-parameters, reset to standard ones')

            Xnn = gp_nnjoint.nn.np_forward(self.Xnorm)
            # Xnn = gp_nnjoint.nn.np_forward(self.X_inf)
            self.Xnn = Xnn
            self.hyps.append(self.get_hyps(gp_nnjoint))

            # mgp = gpflow.models.GPR(X=np.copy(Xnn), Y=np.copy(self.Ynorm), kern=k_list[1])
            # mgp.likelihood.variance = gp_nnjoint.likelihood.variance.value
            # lik_mgp = mgp.compute_log_likelihood()
            # fmean_test, fvar_test = mgp.predict_f(Xnn)
            # lik_gpnnjoint = gp_nnjoint.compute_log_likelihood()
            # fmean_f, fvar_f = gp_nnjoint.predict_f(Xnn)
            # err_lik = np.abs(lik_mgp - lik_gpnnjoint)
            # err_posterior_mean = np.max(np.abs(fmean_test - fmean_f))
            # err_mean = np.max(np.abs(fmean_f - self.Ynorm))
            # err_posterior_var = np.max(np.abs(fvar_test - fvar_f))
            # Xnew = np.tile(np.linspace(start=0., stop=1., num=500)[:, None], [1, Xnn.shape[1]])
            # uZ, uZ_test, prior_meanXnn, uZnew = gp_nnjoint.test_log_likelihood(Xnew)

            # optimize the acquisition function within 0,1 bounds
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition(loss=self.loss, gpmodel=gp_nnjoint, **kwargs)

            if (j%10 == 0):
                self.learnLipschitz(gp_nnjoint)     # learn new Lipschitz constant every 10 iters

            opt_config = self.NLconstraint(opt_config)
            x_proj_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)

            x_tp1 = self.generate_x(x_proj_tp1, gp_nnjoint)

            y_tp1 = self.evaluate(x_tp1)
            self.update_data(x_tp1, y_tp1)
        lik = []

        return self.data_x, self.data_y, self.hyps, lik

    def minimize_acquisition(self, acquisition, opt_config):

        # acquisition_grid, acq_sum, acq_grad = acquisition(self.latent_grid)
        # indices_sorted = np.argsort(acquisition_grid, axis=0)

        KL = self.KLdist(self.grid, self.Xnn)                   # M x N
        KLmin = np.amin(KL, axis=1, keepdims=True)
        indices_sorted = np.argsort(KLmin, axis=0)
        # indices_sorted = np.random.choice(KLmin.shape[0], KLmin.shape[0], replace=False)[:, None]
        x_topk = np.ravel(np.copy(self.grid[indices_sorted[:self.num_init, 0], :]))     # check copy necessary

        def acq_objective(self, xopt, acquisition):
            xopt_reshape = np.reshape(xopt, [self.num_init, self.proj_dim])
            acq_arr, acq_sum, acq_grad = acquisition(xopt_reshape)
            grad_reshape = np.ravel(acq_grad[0])   # check shape
            return  acq_sum[0, 0], grad_reshape
        sum_acquisition_opt = lambda x: acq_objective(self=self, xopt=x, acquisition=acquisition)
        optimize_result = minimize(sum_acquisition_opt, x_topk, **opt_config)

        x_opt_all = np.reshape(optimize_result.x, newshape=[self.num_init, self.proj_dim])
        f_opt_all, f_sum, f_grad = acquisition(x_opt_all)
        x_opt = x_opt_all[f_opt_all.argmin(), :][None]
        f_opt = f_opt_all.min()[None, None]
        return x_opt, f_opt

    def evaluate(self, x_tp1):
        return self.objective.f(x_tp1, noisy=True, fulldim=self.identity)

    def update_data(self, x_new, y_new):
        self.data_x = np.concatenate([self.data_x, x_new], axis=0)     # check shapes
        self.data_y = np.concatenate([self.data_y, y_new], axis=0)
        self.initialize_X()

    def get_hyps_mo(self, gp):
        lengthscales = gp.kern.kernels[0].lengthscales.read_value()
        kern_var = gp.kern.kernels[0].variance.read_value()[None]
        noise_var = gp.likelihood.variance.read_value()[None]
        return np.concatenate([lengthscales, kern_var, noise_var], axis=0)

    def reset_hyps_mo(self, gp):
        gp.kern.kernels[0].lengthscales = np.ones(shape=[self.proj_dim])  # check shape, check transformation hyps
        gp.kern.kernels[0].variance = 1.
        gp.likelihood.variance = 1.
        # gp.read_trainables()
        return gp

    def reset_hyps(self, gp):
        gp.kern.lengthscales = np.ones(shape=[self.proj_dim])  # check shape, check transformation hyps
        gp.kern.variance = 1.
        gp.likelihood.variance = 1.
        # gp.read_trainables()
        return gp

    def get_hyps(self, gp):
        lengthscales = gp.kern.kernels[-1].lengthscales.read_value()
        kern_var = gp.kern.kernels[-1].variance.read_value()[None]
        noise_var = gp.likelihood.variance.read_value()[None]
        return np.concatenate([lengthscales, kern_var, noise_var], axis=0)