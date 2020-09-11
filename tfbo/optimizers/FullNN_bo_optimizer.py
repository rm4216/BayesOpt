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
from scipy.stats import norm


class FullNN_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        # self.initialize_normalization()  # check correct shapes
        self.initialize_probit()
        self.identity = False
        self.Mo_dim = self.Xprobit.shape[1]
        self.Xnn = None
        self.latent_bound = []
        self.latent_grid = []
        self.L = 1.


    def initialize_modelM(self):


        nn = Ort_NN(dims=[self.Xprobit.shape[1], 20, self.proj_dim], N=0, proj_dim=0,
                    name=None)

        k_list, gp_nnjoint = initialize_m_models(x=np.copy(self.Xprobit), y=np.copy(self.Ynorm),
                                                 input_dim=self.proj_dim,
                                                 model='joint_Full',
                                                 kernel='Matern52',
                                                 ARD=True,
                                                 nn=nn)     # last kernel is the Manifold GP kernel

        gp_nnjoint.likelihood.variance = 1e-06  # 0.001

        return k_list, gp_nnjoint, nn


    def generate_x(self, x_proj, gp_joint):

        mean_new, cov_new = gp_joint.predict_x(x_proj)
        joint_new = np.transpose(mean_new)
        # coregion_test = gpflow.kernels.Coregion(input_dim=1, output_dim=int(60), rank=int(60), active_dims=[int(10)])
        # coregion_test.W = np.copy(gp_joint.kern.kernels[0].W.read_value())
        # coregion_test.kappa = np.copy(gp_joint.kern.kernels[0].kappa.read_value())
        # k_test = gpflow.kernels.Matern52(input_dim=int(10), ARD=True, active_dims=list(range(int(10))),
        #                                  lengthscales=np.ones(shape=[int(10)]) * 0.2)
        # k_test.variance = gp_joint.kern.kernels[1].variance.read_value()
        # k_test.lengthscales = np.copy(gp_joint.kern.kernels[1].lengthscales.read_value())
        # kern_test = k_test * coregion_test
        # Xmogp = np.concatenate(
        #     [np.concatenate([np.copy(self.Xnn), np.ones(shape=[self.Xnn.shape[0], 1]) * i], axis=1) for i in
        #      range(self.Mo_dim)], axis=0)
        # Ymogp = np.reshape(np.transpose(self.Xprobit), newshape=[self.Mo_dim * self.Xnn.shape[0], 1])
        # mogp = gpflow.models.GPR(X=Xmogp, Y=Ymogp, kern=kern_test)
        # mogp.likelihood.variance = gp_joint.likelihood.variance.read_value()
        # X_proj_test = np.concatenate(
        #     [np.concatenate([np.copy(x_proj), np.ones(shape=[x_proj.shape[0], 1]) * i], axis=1) for i in
        #      range(self.Mo_dim)], axis=0)
        # fmean_mogp_test, fvar_mogp_test = mogp.predict_f(X_proj_test)
        # err_fmean = np.max(np.abs(mean_new - fmean_mogp_test))
        # err_fvar = np.max(np.abs(np.diag(cov_new)[:, None] - fvar_mogp_test))
        joint_var = np.diag(cov_new)[None]
        joint_var = np.clip(np.abs(joint_var), a_min=1e-06, a_max=1e09)
        m = 0.
        v = 1.
        z = (joint_new - m) / (v * np.sqrt(1. + joint_var / (v ** 2.)))                    # Equation (3.82) GPML book = Equation (3.25) GPML book
        x_out = norm.cdf(z)

        if np.isnan(x_out.max()):
            print('joint_new')
            print(joint_new)
            print('joint_var')
            print(joint_var)
            print('z')
            print(z)
            print('mean_new')
            print(mean_new)
            print('cov_new')
            print(cov_new)
            print('x_out')
            print(x_out)
            print('alpha_reshape')
        # x_new = (joint_new * self.X_std) + self.X_mean     # originally mapped to Xnorm
        # x_out = np.clip(x_out, a_min=0., a_max=1.)
        return x_out


    def update_latent_bounds(self, opt_config):
        m_min = np.clip(np.min(self.Xnn, axis=0, keepdims=True) - 0.2, a_min=0., a_max=1.)  # refined
        m_max = np.clip(np.max(self.Xnn, axis=0, keepdims=True) + 0.2, a_min=0., a_max=1.)
        self.latent_bound = []
        for i in range(self.proj_dim):
            self.latent_bound += [(m_min[0, i].copy(), m_max[0, i].copy())]
        # # uncomment if want to use L-BFGS-B with bounds on the optimization variable
        opt_config['bounds'] = self.latent_bound * self.num_init

        self.latent_grid = np.multiply(np.copy(self.grid), m_max - m_min) + m_min
        return opt_config

    def run(self, maxiters=20):
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')
        for j in range(maxiters):
            print('iteration: ', j)

            self.reset_graph()

            # initialize model
            k_list, gp_nnjoint, nn = self.initialize_modelM()
            gp_nnjoint.kern.kernels[1].variance = 10.

            try:
                gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
            except:
                try:
                    gp_nnjoint.likelihood.variance = 1e-03
                    gp_nnjoint.kern.kernels[1].lengthscales = np.ones(shape=[self.proj_dim])
                    gpflow.train.ScipyOptimizer().minimize(gp_nnjoint)
                except:
                    print('Failure in optimization of hyper-parameters, reset to standard ones')

            Xnn = gp_nnjoint.nn.np_forward(self.Xprobit)
            self.Xnn = Xnn
            self.hyps.append(self.get_hyps(gp_nnjoint))

            # # test Manifold GP predictions
            # fmean, fvar = gp_nnjoint.predict_f(Xnn)


            # # A bit of testing
            # # fmeanX, mean, y_vec, Xnorm_out, Xnorm_test = gp_nnjoint.predict_x(Xnn)
            # fmeanX = gp_nnjoint.predict_x(Xnn)
            # sampleX = gp_nnjoint.sample_x(Xnn)
            # # vec_Xnorm = np.copy(y_vec)
            # # vec_indices = np.concatenate([np.ones(shape=[Xnn.shape[0], 1]) * i for i in range(self.Mo_dim)], axis=0)
            # # Ymogp = vec_Xnorm
            # Ymogp = np.reshape(np.transpose(self.Xprobit), newshape=[self.Mo_dim * Xnn.shape[0], 1])
            # Xmogp = np.concatenate(
            #     [np.concatenate([np.copy(Xnn), np.ones(shape=[Xnn.shape[0], 1]) * i], axis=1) for i in
            #      range(self.Mo_dim)], axis=0)
            # kernmogp = k_list[0] * k_list[1]
            # mogp = gpflow.models.GPR(X=Xmogp, Y=Ymogp, kern=kernmogp)
            # mogp.likelihood.variance = gp_nnjoint.likelihood.variance.value
            # fmean_mogp, fvar_mogp = mogp.predict_f(Xmogp)
            # fmean_mogp0 = np.reshape(np.copy(fmean_mogp[:, 0]), newshape=[self.Mo_dim, Xnn.shape[0]]).transpose()
            # err_fmean = np.max(np.abs(fmean_mogp0 - self.Xprobit))
            # lik_mogp = mogp.compute_log_likelihood()
            # lik_gpnnjoint = gp_nnjoint.compute_log_likelihood()
            # K_test, B_test, y_vec_test, l_k_test, q_k_test, l_b_test, q_b_test, QbQkX_vec_test, kron_diag_test, Inv_vec_test, alpha_test = gp_nnjoint.test_log_likelihood()
            # Knn = gp_nnjoint.kern.kernels[1].compute_K_symm(Xnn)
            # err_K = np.max(np.abs(Knn - K_test))
            # Bnn = np.matmul(gp_nnjoint.kern.kernels[0].W.value,
            #                 np.transpose(gp_nnjoint.kern.kernels[0].W.value)) + np.diag(
            #     gp_nnjoint.kern.kernels[0].kappa.value)
            # err_B = np.max(np.abs(Bnn - B_test))
            # X_vec = np.reshape(np.transpose(self.Xprobit), newshape=[self.Mo_dim * self.Xprobit.shape[0], 1])
            # l_k, q_k = np.linalg.eigh(Knn)
            # l_b, q_b = np.linalg.eigh(Bnn)
            #
            # def mat_vec_mul(B, K, X_vec):
            #     Gb = np.shape(B)[0]
            #     Gk = np.shape(K)[1]
            #     X_Gk = np.reshape(X_vec, newshape=[Gb, Gk])
            #     Z = np.matmul(X_Gk, np.transpose(K))
            #     Z_vec = np.reshape(np.transpose(Z), newshape=[Gb * Gk, 1])
            #     Z_Gb = np.reshape(Z_vec, newshape=[Gk, Gb])
            #     M = np.matmul(Z_Gb, np.transpose(B))
            #     x_out = np.reshape(np.transpose(M), newshape=[-1, 1])
            #     return x_out
            #
            # QbQkX_vec = mat_vec_mul(np.transpose(q_b), np.transpose(q_k), X_vec)
            # kron_diag = np.concatenate([l_k[:, None] * l_b[i] for i in range(self.Mo_dim)], axis=0)
            # Inv_vec = QbQkX_vec / (kron_diag + gp_nnjoint.likelihood.variance.value)
            # alpha_gpnnjoint = mat_vec_mul(q_b, q_k, Inv_vec)
            # err_alpha = np.max(np.abs(alpha_gpnnjoint - alpha_test))
            # logpdf_gpnnjoint = -0.5 * np.matmul(np.transpose(X_vec), alpha_gpnnjoint) - 0.5 * X_vec.shape[0] * np.log(
            #     2 * np.pi) - 0.5 * np.sum(np.log(kron_diag + gp_nnjoint.likelihood.variance.value))
            # err_ll = np.abs(logpdf_gpnnjoint - lik_mogp)
            # mgp = gpflow.models.GPR(X=np.copy(Xnn), Y=np.copy(self.Ynorm), kern=k_list[2])
            # mgp.likelihood.variance = gp_nnjoint.likelihood.variance.value
            # lik_mgp = mgp.compute_log_likelihood()
            # err_lik_all = np.abs(lik_gpnnjoint - (lik_mogp + lik_mgp))
            # fmean_gpnn, fvar_gpnn = gp_nnjoint.predict_f(Xnn)
            # fmean_mgp, fvar_mgp = mgp.predict_f(Xnn)
            # err_predict_f = np.maximum(np.max(np.abs(fmean_gpnn - self.Ynorm)), np.max(np.abs(fmean_gpnn - fmean_mgp)))
            # err_predict_var = np.max(np.abs(fvar_gpnn - fvar_mgp))
            # err_fmeanX = np.max(np.abs(fmeanX - self.Xprobit))
            # err_sampleX = np.maximum(np.max(np.abs(sampleX - self.Xprobit)), np.max(np.abs(sampleX - fmeanX)))

            # optimize the acquisition function within 0,1 bounds
            kwargs = {'ymin': self.Ynorm.min()}
            acquisition = initialize_acquisition(loss=self.loss, gpmodel=gp_nnjoint, **kwargs)

            opt_config = self.update_latent_bounds(opt_config)
            x_proj_tp1, acq_tp1 = self.minimize_acquisition(acquisition, opt_config)

            x_tp1 = self.generate_x(x_proj_tp1, gp_nnjoint)

            y_tp1 = self.evaluate(x_tp1)
            self.update_data(x_tp1, y_tp1)
        lik = []

        return self.data_x, self.data_y, self.hyps, lik

    def minimize_acquisition(self, acquisition, opt_config):

        acquisition_grid, acq_sum, acq_grad = acquisition(self.latent_grid)
        indices_sorted = np.argsort(acquisition_grid, axis=0)
        x_topk = np.ravel(np.copy(self.latent_grid[indices_sorted[:self.num_init, 0], :]))

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
        # self.initialize_normalization()
        self.initialize_probit()

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