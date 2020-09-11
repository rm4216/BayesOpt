import numpy as np
from tfbo.optimizers.optimizer_class import optimizer


class random_optimizer(optimizer):
    def __init__(self, xy_start, proj_dim, objective, loss, seed, **kwargs):
        super().__init__(xy_start, proj_dim, objective, loss)
        self.seed = seed
        self.log_lik_opt = []

    def run(self, maxiters):
        np.random.seed(self.seed)
        N = self.data_x.shape[0]
        shape_solutions = [maxiters + N, self.input_dim]
        x_opt = np.random.uniform(low=0., high=1., size=np.prod(shape_solutions)).reshape(shape_solutions)
        x_opt_append = np.copy(x_opt[N:, :])
        y_opt_append = self.objective.f(x_opt_append, noisy=True, fulldim=False)
        self.update_data(x_opt_append, y_opt_append)
        assert np.max(np.abs(self.data_x - x_opt)) < 1e-12  # x_opt = [data_x_init; x_opt_append]
        self.hyps = np.zeros(shape=[maxiters, int(1)])
        self.log_lik_opt = np.zeros(shape=[maxiters, int(1)])
        return self.data_x, self.data_y, self.hyps, self.log_lik_opt

    def update_data(self, xin, yin):
        self.data_x = np.concatenate([self.data_x, xin], axis=0)
        self.data_y = np.concatenate([self.data_y, yin], axis=0)