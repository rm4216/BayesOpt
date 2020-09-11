import numpy as np
from tfbo.optimizers.optimizer_class import optimizer
from tfbo.models.square_distances import square_dists_np
from tfbo.components.initializations import initialize_models, initialize_acquisition
import gpflow
from tfbo.utils.import_modules import import_attr


class manual_bo_optimizer(optimizer):
    def __init__(self,  xy_start, proj_dim, objective, loss, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.decomposition = [np.arange(start=i * self.proj_dim, stop=(i+1) * self.proj_dim) for i in
                              range(int(np.floor(self.input_dim/self.proj_dim)))]
        self.lipschitz_const = self.objective.lipschitz_const   # 1e01     # check with bin_size
        self.initialize_normalization()
        self.log_lik_opt = []
        self.hyps_opt = []

    def remove_inconsistencies(self, _x, _y):
        indices_opt = list(range(len(self.decomposition)))
        select_component_i = lambda index_j: np.copy(_x[:, self.decomposition[index_j]])
        Xselected = list(map(select_component_i, indices_opt))

        euclidean_dists2y = square_dists_np(_y, _y)
        def select_consistent(Xselected_i):
            euclidean_dists2x = square_dists_np(Xselected_i, Xselected_i)    # double check with tf implementation
            indices_cons_i = self.select_data_indices(euclidean_dists2x, euclidean_dists2y, _y)
            return np.copy(Xselected_i[indices_cons_i, :]), np.copy(_y[indices_cons_i, :])
        XY_cons = list(map(select_consistent, Xselected))
        _Xcons = [XY_cons_i[0] for XY_cons_i in XY_cons]
        _Ycons = [XY_cons_i[1] for XY_cons_i in XY_cons]
        return _Xcons, _Ycons

    def select_data_indices(self, square_dists_x, square_dists_y, _y):
        assert np.max(np.abs(np.diag(square_dists_x) - np.diag(square_dists_y))) < 1e-12
        Lp = np.sqrt(square_dists_y) > self.lipschitz_const * np.sqrt(square_dists_x)   # violate lipschitz continuity -> 1 = inconsistencies
        i_triangle, j_triangle = np.triu_indices(n=square_dists_x.shape[0], k=1)        # collect upper triangular indices

        Lp_triang = np.copy(Lp[i_triangle, j_triangle])

        if np.any(Lp_triang):
            i_triangle_true = np.copy(i_triangle[Lp_triang])[:, None]
            j_triangle_true = np.copy(j_triangle[Lp_triang])[:, None]
            indices_pairs_inc = np.concatenate([i_triangle_true, j_triangle_true], axis=1)  # i,j indices of inconsistent pairs

            pw_max = lambda ij: self.select_max(ind_ij=ij, _Y=_y)
            indices_to_remove_list = list(np.ravel(list(map(pw_max, list(indices_pairs_inc)))))
            consistent_selection = np.ones(shape=square_dists_x.shape[0], dtype=bool)
            if indices_to_remove_list:  # empty sequences are false, if nonempty
                indices_to_remove_norep = list(set(indices_to_remove_list))     # self.remove_reps(indices_to_remove_list) remove repeated indices in list
                consistent_selection[indices_to_remove_norep] = False           # !indices_to_remove_norep have lost order!
        else:
            consistent_selection = np.ones(shape=square_dists_x.shape[0], dtype=bool)
        return consistent_selection

    def remove_reps(self, indices_list):
        bag_of_indices = np.copy(np.array([indices_list[0]]))[:, None]  # why [0]
        for i in range(len(indices_list) - 1):
            index_new = indices_list[i+1]
            if all([index_i != index_new for index_i in list(bag_of_indices)]):
                bag_of_indices = np.concatenate([bag_of_indices, np.copy(np.array([index_new]))[:, None]], axis=0)
            i += int(1)
        return bag_of_indices

    def select_max(self, ind_ij, _Y):
        # _Y = self.data_y
        if _Y[ind_ij[0]] >= _Y[ind_ij[1]]:
            return np.array([np.copy(ind_ij[0])])
        else:
            return np.array([np.copy(ind_ij[1])])

    def initialize_single_model(self, x_consist_i, y_consist_i):
        kernel_out, model_out = initialize_models(x=np.copy(x_consist_i), y=np.copy(y_consist_i),
                                                  input_dim=self.proj_dim, model='GPR', kernel='Matern52', ARD=True)
        return kernel_out, model_out

    def compose_x(self, x_list):
        x0 = np.zeros(shape=[x_list[0].shape[0], self.proj_dim * len(x_list)])  # check shapes of x_i
        for x_i, indices_i in zip(x_list, self.decomposition):
            x0[:, indices_i] = x_i  # if I change x_i changes x0?
        return x0

    def evaluate(self, x_list):
        x_tp1 = self.compose_x(x_list)  # check shape
        y_tp1 = self.objective.f(x_tp1, noisy=True, fulldim=False)
        return y_tp1

    def fit_gp(self, gp):
        self.hyps_opt = []
        signal_variance = np.random.uniform(low=0., high=10, size=10)[:, None]
        noise_variance = signal_variance * 0.01
        log_lik = []
        for signal_i, noise_i in zip(list(signal_variance), list(noise_variance)):
            gp = self.reset_hyps(gp)
            gp.kern.variance = signal_i[0]
            gp.likelihood.variance = noise_i[0]
            try:
                gpflow.train.ScipyOptimizer().minimize(gp)
            except:
                gp = self.reset_hyps(gp)
            log_lik.append(gp.compute_log_likelihood())
            self.hyps_opt.append(self.get_hyps(gp))

        np_log_liks = np.array(log_lik)
        index_opt = np.argmax(np_log_liks)
        gp = self.assign_hyps(gp, self.hyps_opt[index_opt])
        self.log_lik_opt.append(np_log_liks[index_opt])
        return gp

    def optimize_ith_model(self, gp_i, i, opt_config):
        # gp_i = self.reset_hyps(gp_i)
        gp_i = self.fit_gp(gp_i)
        # try:
        #     gpflow.train.ScipyOptimizer().minimize(gp_i)
        # except:
        #     gp_i = self.reset_hyps(gp_i)
        self.hyps.append(self.get_hyps(gp_i))

        kwargs = {'ymin': self.Ynorm.min()}
        acquisition = initialize_acquisition(self.loss, gp_i, **kwargs)
        acquisition_norm = lambda x: \
            self.acquisition_norm(acquisition=acquisition, x=x,
                                  X_proj_mean=np.copy(self.X_mean[:, self.decomposition[i]]),
                                  X_proj_std=np.copy(self.X_std[:, self.decomposition[i]]))
        x_opt, acq_opt = self.minimize_acquisition(acquisition_norm, opt_config)
        return x_opt, acq_opt

    def update_data(self, xnew, ynew):
        self.data_x = np.concatenate([self.data_x, xnew], axis=0)
        self.data_y = np.concatenate([self.data_y, ynew], axis=0)
        self.initialize_normalization()

    def update_models(self, list_gp, list_x, list_y):
        for gp_i, x_i, y_i in zip(list_gp, list_x, list_y):
            gp_i.X = np.copy(x_i)
            gp_i.Y = np.copy(y_i)
        return list_gp

    def run(self, maxiters):
        # Normalization -> decomposition -> initialization/update of each GP model
        list_Xnorm_cons, list_Ynorm_cons = self.remove_inconsistencies(_x=np.copy(self.Xnorm), _y=np.copy(self.Ynorm))
        list_km_out = list(map(self.initialize_single_model, list_Xnorm_cons, list_Ynorm_cons))
        # list_kernels = [km_i[0] for km_i in list_km_out]
        list_gpmodels = [km_i[1] for km_i in list_km_out]
        opt_config = import_attr('tfbo/configurations', attribute='acquisition_opt')
        opt_config['bounds'] = [(0., 1.)] * self.proj_dim * self.num_init

        list_components = list(range(len(self.decomposition)))
        optimize_i = lambda gp_i, i: self.optimize_ith_model(gp_i, i, opt_config)

        for j in range(maxiters):
            print(j)

            list_xa = list(map(optimize_i, list_gpmodels, list_components))  # check double input

            list_x = [xa_i[0] for xa_i in list_xa]
            x_out = self.compose_x(list_x)
            y_out = self.evaluate(list_x)
            self.update_data(xnew=x_out, ynew=y_out)    # augment dataset and normalize
            list_Xnorm_cons, list_Ynorm_cons = self.remove_inconsistencies(_x=np.copy(self.Xnorm), _y=np.copy(self.Ynorm))  # decompose and prune
            self.reset_graph()
            list_km_out = list(map(self.initialize_single_model, list_Xnorm_cons, list_Ynorm_cons))
            list_gpmodels = [km_i[1] for km_i in list_km_out]
            # list_gpmodels = self.update_models(list_gp=list_gpmodels, list_x=list_Xnorm_cons, list_y=list_Ynorm_cons)
        return self.data_x, self.data_y, self.hyps, self.log_lik_opt