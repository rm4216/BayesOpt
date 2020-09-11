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
from tfbo.models.cov_funcs import LinearGeneralized
from gpflow.conditionals import base_conditional


class SampleBLRNN_bo_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.initialize_X()  # check correct shapes
        self.identity = False
        self.Mo_dim = self.data_x.shape[1]
        self.Xnn = None
        self.latent_bound = []
        self.latent_grid = []
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

        # x_new = (mean_new * self.X_std) + self.X_mean
        x_new = (mean_new * self.X_std) + self.X_mean     # originally mapped to Xnorm
        # x_out = np.clip(x_new, a_min=0., a_max=1.)
        # return x_out
        # x_new = np.copy(mean_new)
        return self.sigmoid(x_new)          # check shape


    def initialize_X(self):

        self.Y_mean = np.mean(self.data_y, axis=0, keepdims=True)
        self.Y_std = np.std(self.data_y, axis=0, keepdims=True)

        self.X_inf = self.logit(self.data_x)
        self.X_mean = np.mean(self.X_inf, axis=0, keepdims=True)
        self.X_std = np.std(self.X_inf, axis=0, keepdims=True)

        self.Xnorm = (self.X_inf - self.X_mean) / self.X_std
        # self.X_mean = np.mean(self.data_x, axis=0, keepdims=True)
        # self.X_std = np.std(self.data_x, axis=0, keepdims=True)
        #
        # self.Xnorm = (self.data_x - self.X_mean) / self.X_std
        self.Ynorm = (self.data_y - self.Y_mean) / self.Y_std   # check all shapes

        if np.isnan(self.Ynorm).any(): raise ValueError('Ynorm contains nan')
        if np.isnan(self.Xnorm).any(): raise ValueError('Xnorm contains nan')
        def test_(input_norm):
            xm_test = np.mean(input_norm, axis=0, keepdims=True)
            xst_test = np.std(input_norm, axis=0, keepdims=True) - np.ones([1, input_norm.shape[1]])
            boolm_list = [np.abs(xm_i[0]) <= 1e-02 for xm_i in list(xm_test.transpose())]
            bools_list = [np.abs(xs_i[0]) <= 1e-02 for xs_i in list(xst_test.transpose())]
            assert all(boolm_list) and all(bools_list)
        test_(self.Xnorm)
        test_(self.Ynorm)


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
            gp_nnjoint.kern.kernels[0].variance = 1.

            # # W_0_bt = np.copy(gp_nnjoint.nn.W_0.read_value())
            # # W_1_bt = np.copy(gp_nnjoint.nn.W_1.read_value())
            # # b_0_bt = np.copy(gp_nnjoint.nn.b_0.read_value())
            # # b_1_bt = np.copy(gp_nnjoint.nn.b_1.read_value())
            # # X_mean = np.mean(self.X_inf, axis=0, keepdims=True)
            # # X_std = np.std(self.X_inf, axis=0, keepdims=True)
            # # Xnorm = (self.X_inf - X_mean) / X_std
            # St = (1 / self.Xnorm.shape[0]) * np.matmul(self.Xnorm.transpose(), self.Xnorm)
            # l_St, q_St = np.linalg.eigh(St)
            # assert np.max(np.abs(np.matmul(np.matmul(q_St, np.diag(l_St)), q_St.transpose()) - St)) < 1e-09
            # assert np.max(np.abs(np.eye(q_St.shape[0]) - np.matmul(q_St, q_St.transpose()))) < 1e-09
            # assert np.max(np.abs(np.eye(q_St.shape[0]) - np.matmul(q_St.transpose(), q_St))) < 1e-09
            # U_d = np.copy(q_St[:, -self.proj_dim:])
            # max_evals = np.copy(l_St[-self.proj_dim:][None])
            # # assert np.max(np.abs(np.matmul(St, U_d) / max_evals - U_d)) < 1e-09
            # Y_d = np.matmul(U_d.transpose(), self.Xnorm.transpose())
            # gp_nnjoint.nn.W_0 = np.copy(U_d.transpose())

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

            # W_0_at = np.copy(gp_nnjoint.nn.W_0.read_value())
            # W_1_at = np.copy(gp_nnjoint.nn.W_1.read_value())
            # b_0_at = np.copy(gp_nnjoint.nn.b_0.read_value())
            # b_1_at = np.copy(gp_nnjoint.nn.b_1.read_value())
            #
            # mgp = gpflow.models.GPR(X=np.copy(Xnn), Y=np.copy(self.Ynorm), kern=k_list[1])
            # mgp.likelihood.variance = np.copy(gp_nnjoint.likelihood.variance.value) + np.copy(gp_nnjoint.jitter.read_value())
            # lik_mgp = mgp.compute_log_likelihood()
            # fmean_test, fvar_test = mgp.predict_f(Xnn)
            # lik_gpnnjoint = gp_nnjoint.compute_log_likelihood()
            # lik_gpnnjoint0 = gp_nnjoint.compute_log_likelihood()
            # err_repeat_lik = np.abs(lik_gpnnjoint - lik_gpnnjoint0)
            # fmean_f, fvar_f = gp_nnjoint.predict_f(Xnn)
            # # err_lik = np.abs(lik_mgp - lik_gpnnjoint)
            # err_posterior_mean = np.max(np.abs(fmean_test - fmean_f))
            # err_mean = np.max(np.abs(fmean_f - self.Ynorm))
            # err_posterior_var = np.max(np.abs(fvar_test - fvar_f))
            # Xnew = np.tile(np.linspace(start=0., stop=1., num=500)[:, None], [1, Xnn.shape[1]])
            # logpdf_test, logpdfm_test, uZ, uZ_test_reshape, uZnew, Kprior_inv, Kprior, check_same_as_X_vec, X_vec, SMarg_inv, SMarg, Vprior_inv, Vprior, LK_invT, logpdfm_test_eigh, e_Km, v_Km, Km_test, d_eigh, Km_component, Lm_component = gp_nnjoint.test_log_likelihood(Xnew)
            # logpdf_test0, logpdfm_test0, uZ0, uZ_test_reshape0, uZnew0, Kprior_inv0, Kprior0, check_same_as_X_vec0, X_vec0, SMarg_inv0, SMarg0, Vprior_inv0, Vprior0, LK_invT0, logpdfm_test_eigh0, e_Km0, v_Km0, Km_test0, d_eigh0, Km_component0, Lm_component0 = gp_nnjoint.test_log_likelihood(Xnew)
            #
            # # check same values are returned over multiple calls
            # assert logpdf_test == logpdf_test0 and logpdfm_test == logpdfm_test0 and np.all(uZ == uZ0) and np.all(
            #     uZ_test_reshape == uZ_test_reshape0) and np.all(Kprior_inv == Kprior_inv0) and np.all(
            #     Kprior == Kprior0) and np.all(check_same_as_X_vec == check_same_as_X_vec0) and np.all(
            #     X_vec == X_vec0) and np.all(SMarg_inv == SMarg_inv0) and np.all(SMarg == SMarg0) and np.all(
            #     Vprior_inv == Vprior_inv0) and np.all(Vprior == Vprior0) and np.all(LK_invT == LK_invT0)
            # assert np.all(logpdfm_test_eigh == logpdfm_test_eigh0) and np.all(e_Km == e_Km0) and np.all(
            #     v_Km == v_Km0) and np.all(Km_test == Km_test0) and np.all(d_eigh == d_eigh0)
            # assert np.all(Km_component == Km_component0) and np.all(Lm_component == Lm_component0)
            #
            # kernBLR = LinearGeneralized(input_dim=self.Mo_dim, L_p=LK_invT)
            # gpBLR = gpflow.models.GPR(X=np.copy(uZ.transpose()), Y=np.copy(self.X_inf), kern=kernBLR)
            # # gpBLR = gpflow.models.GPR(X=np.copy(uZ.transpose()), Y=np.copy(self.Xnorm), kern=kernBLR)
            # gpBLR.likelihood.variance = np.copy(gp_nnjoint.noise_variance.read_value())
            # lik_BLR = gpBLR.compute_log_likelihood()
            # err_logpdfBLR = np.abs(lik_BLR - logpdf_test)
            #
            # # perform checkings listed in "_test_likelihood" method
            # threshold = 1e-04
            # assert np.all(uZ_test_reshape == uZ) and np.max(
            #     np.abs(np.matmul(Kprior_inv, Kprior) - np.eye(Kprior.shape[0]))) < threshold and np.all(
            #     check_same_as_X_vec == X_vec) and np.max(
            #     np.abs(np.matmul(SMarg_inv, SMarg) - np.eye(SMarg.shape[0]))) < threshold and np.max(
            #     np.abs(np.matmul(Vprior_inv, Vprior) - np.eye(Vprior.shape[0]))) < threshold and np.abs(
            #     logpdfm_test - lik_mgp) < threshold and np.abs(lik_BLR - logpdf_test) < threshold and np.abs(
            #     lik_gpnnjoint - lik_BLR - lik_mgp) < threshold
            #
            # # uZsT_cov = gp_nnjoint.test_predict_x(self.Xnn[0, :][None])
            # reconstructed_x = []
            # for i in range(self.Xnn.shape[0]):
            #     Xmean_i, Spost, V = gp_nnjoint.predict_x(self.Xnn[i, :][None])
            #     reconstructed_x.append(Xmean_i)
            # X_inf_reconstructed = np.concatenate(reconstructed_x, axis=0)
            # err_reconstruction = np.max(np.abs(X_inf_reconstructed - self.X_inf))
            # # err_reconstruction = np.max(np.abs(X_inf_reconstructed - self.Xnorm))
            #
            # LSxx_invT, MNT, post_meanT, post_S_test, Vp_test, uZs_test, uZ_test = gp_nnjoint.test_predict_x(self.Xnn[0, :][None])
            # kernBLR_post = LinearGeneralized(input_dim=self.Mo_dim, L_p=LSxx_invT)
            # mean_function_post = gpflow.mean_functions.Linear(A=MNT, b=np.zeros(1))
            # gpBLR_post = gpflow.models.GPR(X=np.copy(uZ.transpose()), Y=np.copy(self.X_inf), kern=kernBLR_post, mean_function=mean_function_post)
            # # gpBLR_post = gpflow.models.GPR(X=np.copy(uZ.transpose()), Y=np.copy(self.Xnorm), kern=kernBLR_post, mean_function=mean_function_post)
            # gpBLR_post.likelihood.variance = np.copy(gp_nnjoint.noise_variance.read_value())
            # X_inf_rec_test, X_inf_var_test = gpBLR_post.predict_f(np.copy(uZ.transpose()))
            # assert np.max(np.abs(X_inf_reconstructed - X_inf_rec_test)) < 1e-01


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