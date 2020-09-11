import numpy as np
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..', '..'))
from tfbo.utils.load_save import loadMat
import random
random.seed(a=123)


# np.random.seed(123)
class Michalewicz10D(object):
    def __init__(self, m=0.5):
        self.high_dim = int(100)
        self.input_dim = int(10)
        self.relevant_dims = np.array([16, 21, 31, 38, 45, 78, 84, 85, 91, 95], dtype=int)
        self.m = m

        self.minimizer = np.zeros(shape=[1, self.high_dim], dtype=np.float64)
        # self.minimizer[:, self.relevant_dims] = np.array([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796,
        #                                                    1.454414, 1.756087, 1.655717, 1.570796]])/np.pi     # 1x10
        self.minimizer[:, self.relevant_dims] = np.array([[1.9756, 1.5708, 1.3227, 1.1611, 1.0464, 0.9598, 0.8916,
                                                           1.7539, 1.6548, 1.5708]]) / np.pi  # 1x10
        self.fmin = self.f(self.minimizer)  # m=10., fmin=-9.66015; m=0.5, fmin=-9.1415
        self.scale = 0.01
        self.lipschitz_const = 35.45131945027451

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is Nx10 with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x_resc = x * np.pi
        else:
            x_resc = np.copy(x[:, self.relevant_dims]) * np.pi  # Nx10
        i_s = np.arange(self.input_dim)[None] + 1
        fact1 = np.sin(np.multiply(i_s, np.square(x_resc))/np.pi) ** (2 * self.m)
        factr2 = np.sin(x_resc)
        f_x = - np.sum(np.multiply(fact1, factr2), axis=1, keepdims=True)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1
# mic = Michalewicz10D()

class Hartmann6D(object):
    def __init__(self):
        self.high_dim = int(60)
        self.input_dim = int(6)
        self.relevant_dims = np.array([1, 6, 14, 41, 47, 56], dtype=int)
        self.A = np.array([[10., 3., 17., 3.50, 1.7, 8],
                           [0.05, 10., 17., 0.1, 8., 14.],
                           [3., 3.5, 1.7, 10., 17., 8.],
                           [17., 8., 0.05, 10., 0.1, 14.]])
        self.P = 1e-04 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                                   [2329., 4135., 8307., 3736., 1004., 9991.],
                                   [2348., 1451., 3522., 2883., 3047., 6650.],
                                   [4047., 8828., 8732., 5743., 1091., 381.]])
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])

        self.minimizer = np.zeros(shape=[1, self.high_dim], dtype=np.float64)
        self.minimizer[:, self.relevant_dims] = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[None]
        self.fmin = self.f(self.minimizer)  # -3.32237
        self.scale = 0.01

    def f(self, x, noisy=False, fulldim=False):
        '''
        x is assumed 6-dimensional between [0, 1], shape Nx6
        :param x: Nx6
        :return: f_x(x) Nx1
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x = x
        else:
            x = np.copy(x[:, self.relevant_dims])
        val_i = []
        for x_i in list(x):     # x_i 1x6
            exponents = np.sum(np.multiply(np.square(self.P - x_i), self.A), axis=1)
            val_i.append(-np.sum(np.multiply(np.exp(-exponents), self.alpha), axis=0))
        f_x = np.vstack(val_i)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

class ProductSines10D(object):
    def __init__(self, multip=1.):
        self.high_dim = int(100)
        self.input_dim = int(10)
        self.relevant_dims = np.array([16, 21, 31, 38, 45, 78, 84, 85, 91, 95], dtype=int)
        # self.relevant_dims = np.array([16, 21, 31, 38], dtype=int)

        # self.minimizer = np.ones(shape=[1, self.high_dim], dtype=np.float64) * (np.pi * (3. / 2.)) / (2*np.pi)
        self.minimizer = np.ones(shape=[1, self.high_dim], dtype=np.float64) * ((np.pi * (3. / 2.)) - np.pi )/ (np.pi)
        self.fmin = self.f(self.minimizer)
        self.scale = 0.01
        self.lipschitz_const = 55.04895709974139

    def f(self, x, noisy=False, fulldim=False):
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x_resc = x * np.pi + np.pi # (2. * np.pi)
        else:
            x_resc = np.copy(x[:, self.relevant_dims]) * np.pi + np.pi # (2. * np.pi)    # [0, 1] -> [0, 2pi]
        f_x = np.sin(x_resc[:, 0])[:, None] * np.prod(np.sin(x_resc), axis=1)[:, None] * 10           # rescale by factor
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # + np.random.normal(loc=0., scale=0.1, size=1)

class Shekel4D(object):
    def __init__(self):
        self.high_dim = int(40)
        self.input_dim = int(4)
        self.relevant_dims = np.array([8, 10, 19, 33], dtype=int)
        self.m = 10.
        self.beta = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])[:, None]
        self.C = np.array([[4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                           [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6],
                           [4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                           [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6]])

        self.minimizer = np.zeros(shape=[1, self.high_dim])
        self.minimizer[:, self.relevant_dims] = np.array([4., 4., 4., 4.])[None] / 10.
        self.fmin = self.f(self.minimizer)
        self.scale = 0.01

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming input in [0, 1]^D
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            rescaled_x = 10. * x
        else:
            rescaled_x = 10. * np.copy(x[:, self.relevant_dims])     # input evaluated in [0, 10]
        x_Cdiff2 = np.square(rescaled_x[:, None, :]-np.transpose(self.C)[None])
        inner_sum = np.sum(x_Cdiff2, axis=2) + self.beta.transpose()
        outer_sum = np.sum(1./inner_sum, axis=1)
        f_x = - outer_sum[:, None]
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x

class Branin2D(object):
    def __init__(self):
        self.high_dim = int(20)
        self.input_dim = int(2)
        self.relevant_dims = np.array([4, 14])
        self.a = 1.
        self.b = 5.1 / (4 * np.pi ** 2)
        self.c = 5. / np.pi
        self.r = 6.
        self.s = 10.
        self.t = 1 / (8 * np.pi)

        self.lb0 = -5.
        self.ub0 = 10.
        self.lb1 = 0.
        self.ub1 = 15.

        minimizer = np.zeros(shape=[1, self.high_dim])
        minimizer[:, self.relevant_dims] = self.rescale01(np.array([-np.pi, 12.275])[None])
        self.minimizer = minimizer  # in 20D
        self.minimizer1 = self.rescale01(np.array([np.pi, 2.275])[None])
        self.minimizer2 = self.rescale01(np.array([9.42478, 2.475])[None])
        self.fmin = self.f(self.minimizer)
        self.scale = 0.01

    def f(self, x, noisy=False, fulldim=False):
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x_resc = self.original_scale(x)
        else:
            x_resc = self.original_scale(np.copy(x[:, self.relevant_dims]))  # in [-5, 10]:x1, [0, 15]:x2
        x0 = x_resc[:, 0][:, None]
        x1 = x_resc[:, 1][:, None]
        f_x1 = self.a * (x1 - (self.b * x0 ** 2.) + (self.c * x0) - self.r) ** 2.
        f_x2 = self.s * (1 - self.t) * np.cos(x0)
        f_x = f_x1 + f_x2 + self.s
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x

    def original_scale(self, x_in):
        x_in0 = np.copy(x_in[:, 0])[:, None]
        x_in1 = np.copy(x_in[:, 1])[:, None]
        x0_resc = x_in0 * (self.ub0 - self.lb0) + self.lb0
        x1_resc = x_in1 * (self.ub1 - self.lb1) + self.lb1
        return np.concatenate([x0_resc, x1_resc], axis=1)

    def rescale01(self, x_in):
        x_in0 = np.copy(x_in[:, 0])[None]
        x_in1 = np.copy(x_in[:, 1])[None]
        x0_os = (x_in0 - self.lb0) / (self.ub0 - self.lb0)
        x1_os = (x_in1 - self.lb1) / (self.ub1 - self.lb1)
        return np.concatenate([x0_os, x1_os], axis=1)

# bran = Branin2D()
# x = bran.minimizer
# f_x = bran.f(x)
# print(bran.f(x, noisy=True))



class Rosenbrock10D(object):
    def __init__(self):
        self.high_dim = int(100)
        self.input_dim = int(10)
        self.relevant_dims = np.array([16, 21, 31, 38, 45, 78, 84, 85, 91, 95], dtype=int)

        self.lb = -2.048
        self.ub = 2.048

        self.minimizer = (np.ones(shape=[1, self.high_dim], dtype=np.float64) - self.lb) / (self.ub - self.lb)
        self.fmin = self.f(self.minimizer, fulldim=False, noisy=False)  # fmin = 0
        self.scale = 0.01


    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is Nx10 with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x_resc = x * (self.ub - self.lb) + self.lb                                  # Nx10
        else:
            x_resc = np.copy(x[:, self.relevant_dims]) * (self.ub - self.lb) + self.lb  # Nx10
        xi = x_resc[:, :-1].copy()
        xip1 = x_resc[:, 1:].copy()
        addend = 100. * (xip1 - xi ** 2.) ** 2. + (xi - 1.) ** 2.
        f_x = np.sum(addend, axis=1, keepdims=True)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# ros = Rosenbrock10D()
# x_opt = (np.ones(shape=[5, 10]) - ros.lb) / (ros.ub - ros.lb)
# f_opt = ros.f(x_opt, noisy=False, fulldim=True)
# y_opt = ros.f(x_opt, noisy=True, fulldim=True)
#
# x1 = np.linspace(start=0., stop=1., num=100)[:, None]
# X1, X2 = np.meshgrid(x1, x1)
# X = np.column_stack([np.ravel(X1), np.ravel(X2)])
# f = ros.f(X, noisy=False, fulldim=True)
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X1, X2, np.reshape(f, X1.shape))
# plt.show()



class HartmannLinear6D(object):
    def __init__(self):
        self.high_dim = int(60)
        self.input_dim = int(6)
        self.A = np.array([[10., 3., 17., 3.50, 1.7, 8],
                           [0.05, 10., 17., 0.1, 8., 14.],
                           [3., 3.5, 1.7, 10., 17., 8.],
                           [17., 8., 0.05, 10., 0.1, 14.]])
        self.P = 1e-04 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                                   [2329., 4135., 8307., 3736., 1004., 9991.],
                                   [2348., 1451., 3522., 2883., 3047., 6650.],
                                   [4047., 8828., 8732., 5743., 1091., 381.]])
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])

        self.q, self.q_inv = self.linear_map()
        self.minimizer6 = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[
            None]
        self.fmin6 = self.f(self.minimizer6, noisy=False, fulldim=True)
        self.minimizer = (np.matmul(self.minimizer6 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.         # optimizer in 60-dimensional space in [0,1]
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=False)
        # self.minimizer = np.zeros(shape=[1, self.high_dim], dtype=np.float64)
        # self.minimizer[:, self.relevant_dims] = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[None]
        # self.fmin = self.f(self.minimizer)  # -3.32237
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        # np.save('HartmannLinear6D' + str(self.high_dim) + '_' + str(self.input_dim) + '.npy', Mat)
        # Mat = loadMat('HartmannLinear6D' + str(self.high_dim) + '_' + str(self.input_dim) + '.npy')
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def f(self, x, noisy=False, fulldim=False):
        '''
        x is assumed 60-dimensional between [0, 1], shape Nx60
        :param x: Nx60
        :return: f_x(x) Nx1
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x = x
        else:
            x_m1_1 = x * 2. - 1.                        # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())   # bring to 6-dimensional mostly in [-1, 1]
            x = (np.copy(x) + 1.) / 2.                  # bring to interval [0, 1]
        # here x is assumed 6-dimensional between [0, 1]
        val_i = []
        for x_i in list(x):     # x_i 1x6
            exponents = np.sum(np.multiply(np.square(self.P - x_i), self.A), axis=1)
            val_i.append(-np.sum(np.multiply(np.exp(-exponents), self.alpha), axis=0))
        f_x = np.vstack(val_i)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# hartL = HartmannLinear6D()
# minimizers = np.tile(hartL.minimizer, [5, 1])
# fmins = hartL.f(minimizers, noisy=False, fulldim=False)

# np.random.seed(44)
# Mat = np.random.normal(loc=0., scale=1., size=[hartL.input_dim, hartL.high_dim])
# q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
# q = np.transpose(q)
# x = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[None]
# q_inv = np.linalg.pinv(q)
# minimizer = np.matmul(x, q_inv.transpose())   # minimizer is in the interval [-1, 1] ?
# minimizer01 = (minimizer + 1.) / 2.
# fmin = hartL.f(minimizer01, noisy=False, fulldim=False)



class ProductSinesLinear10D(object):
    def __init__(self):
        self.high_dim = int(60)
        self.input_dim = int(10)

        self.q, self.q_inv = self.linear_map()
        # self.minimizer10 = np.ones(shape=[1, self.input_dim]) * ((np.pi * (3. / 2.)) - np.pi )/ (np.pi)
        self.minimizer10 = np.ones(shape=[1, self.input_dim]) * (np.pi * (3. / 2.))/ (2. * np.pi)
        self.fmin10 = self.f(self.minimizer10, noisy=False, fulldim=True)
        self.minimizer = (np.matmul(self.minimizer10 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=False)
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        # np.save('ProductSinesLinear10D_' + str(self.high_dim) + '_' + str(self.input_dim) + '.npy', Mat)
        # Mat = loadMat('ProductSinesLinear10D_' + str(self.high_dim) + '_' + str(self.input_dim) + '.npy')
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        # q_norm = np.linalg.norm(q, ord=None, axis=1, keepdims=True)
        return q, np.linalg.pinv(q)

    def f(self, x, noisy=False, fulldim=False):
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            # x_resc = x * np.pi + np.pi # (2. * np.pi)   # [0, 1] -> [pi, 2pi]
            x_resc = x * (2. * np.pi)                   # [0, 1] -> [0, 2pi]
        else:
            x_m1_1 = x * 2. - 1.                        # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())   # bring to 10-dimensional mostly [-1, 1]
            x01 = (x + 1.) / 2.                         # bring to interval [0, 1]
            # x_resc = np.copy(x01) * np.pi + np.pi     # (ideally, only true for the optimum) [0, 1] -> [pi, 2pi]
            x_resc = np.copy(x01) * (2. * np.pi)        # (ideally, only true for the optimum) [0, 1] -> [0, 2pi]
        f_x = np.sin(x_resc[:, 0])[:, None] * np.prod(np.sin(x_resc), axis=1)[:, None] * 10           # rescale by factor
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # + np.random.normal(loc=0., scale=0.1, size=1)

# prodL = ProductSinesLinear10D()
# minimizers = np.tile(prodL.minimizer, [5, 1])
# fmins = prodL.f(minimizers, noisy=False, fulldim=False)


class ShekelLinear4D(object):
    def __init__(self):
        self.high_dim = int(60)
        self.input_dim = int(4)
        self.m = 10.
        self.beta = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])[:, None]
        self.C = np.array([[4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                           [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6],
                           [4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                           [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6]])

        self.q, self.q_inv = self.linear_map()
        self.minimizer4 = np.array([4., 4., 4., 4.])[None] / 10.
        self.fmin4 = self.f(self.minimizer4, noisy=False, fulldim=True)     # m=5, fmin=-10.1532;  m=7, fmin=-10.4029;  m=10, fmin=-10.5364
        self.minimizer = (np.matmul(self.minimizer4 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=False)
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming input in [0, 1]^D
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            rescaled_x = 10. * np.copy(x)
        else:
            x_m1_1 = x * 2. - 1.                            # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())       # bring to 4-dimensional # [-1, 1] -> [-1, 1] (ideally, only true for the optimum)
            x = (np.copy(x) + 1.) / 2.                      # bring to interval [0, 1]
            rescaled_x = 10. * np.copy(x)                   # input evaluated in [0, 10] (only true for the optimum)
        x_Cdiff2 = np.square(rescaled_x[:, None, :]-np.transpose(self.C)[None])
        inner_sum = np.sum(x_Cdiff2, axis=2) + self.beta.transpose()
        outer_sum = np.sum(1./inner_sum, axis=1)
        f_x = - outer_sum[:, None]
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x

# shekL = ShekelLinear4D()
# minimizers = np.tile(shekL.minimizer, [5, 1])
# fmins = shekL.f(minimizers, noisy=False, fulldim=False)


class RosenbrockLinear10D(object):
    def __init__(self):
        self.high_dim = int(60)
        self.input_dim = int(10)

        # self.lb = -2.048
        self.lb = -5.
        # self.ub = 2.048
        self.ub = 10.

        self.q, self.q_inv = self.linear_map()
        self.minimizer10 = (np.ones(shape=[1, self.input_dim]) - self.lb) / (self.ub - self.lb)
        self.fmin10 = self.f(self.minimizer10, noisy=False, fulldim=True)
        self.minimizer = (np.matmul(self.minimizer10 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.
        self.fmin = self.f(self.minimizer, fulldim=False, noisy=False)              # fmin = 0
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is Nx10 with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x_resc = x * (self.ub - self.lb) + self.lb                                  # Nx10
        else:
            x_m1_1 = x * 2. - 1.                            # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())       # bring to 10-dimensional # [-1, 1] -> [-1, 1]
            x = (np.copy(x) + 1.) / 2.                      # bring to interval [0, 1]
            x_resc = np.copy(x) * (self.ub - self.lb) + self.lb  # Nx10
        xi = x_resc[:, :-1].copy()
        xip1 = x_resc[:, 1:].copy()
        addend = 100. * (xip1 - xi ** 2.) ** 2. + (xi - 1.) ** 2.
        f_x = np.sum(addend, axis=1, keepdims=True)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# RosL = RosenbrockLinear10D()
# minimizers = np.tile(RosL.minimizer, [5, 1])
# fmins = RosL.f(minimizers, noisy=False, fulldim=False)
# X05 = np.ones(shape=[1, 60]) * 0.5
# f05 = RosL.f(X05, noisy=False, fulldim=False)
# aaa = 5


class MichalewiczLinear10D(object):
    def __init__(self, m=0.5):
        self.high_dim = int(60)
        self.input_dim = int(10)
        self.m = m

        self.q, self.q_inv = self.linear_map()
        self.minimizer10 = np.array([[1.9756, 1.5708, 1.3227, 1.1611, 1.0464, 0.9598, 0.8916,
                                                           1.7539, 1.6548, 1.5708]]) / np.pi
        # self.minimizer10 = np.array([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796,
        #                                                    1.454414, 1.756087, 1.655717, 1.570796]])/np.pi     # m=10
        self.fmin10 = self.f(self.minimizer10, noisy=False, fulldim=True)
        self.minimizer = (np.matmul(self.minimizer10 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=False)  # m=10., fmin=-9.66015; m=0.5, fmin=-9.1415
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is Nx10 with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x_resc = x * np.pi
        else:
            x_m1_1 = x * 2. - 1.                            # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())       # bring to 10-dimensional # [-1, 1] -> [-1, 1]
            x = (np.copy(x) + 1.) / 2.                      # bring to interval [0, 1]
            x_resc = np.copy(x) * np.pi                     # Nx10
        i_s = np.arange(self.input_dim)[None] + 1
        fact1 = np.sin(np.multiply(i_s, np.square(x_resc))/np.pi) ** (2 * self.m)
        factr2 = np.sin(x_resc)
        f_x = - np.sum(np.multiply(fact1, factr2), axis=1, keepdims=True)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# michL = MichalewiczLinear10D()
# minimizers = np.tile(michL.minimizer, [5, 1])
# fmins = michL.f(minimizers, noisy=False, fulldim=False)


# sys.path.append('/vol/bitbucket/rm4216/Desktop/ImperialCollege/MATLAB_API/install/lib/python3.6/site-packages')
# import matlab.engine
# import scipy.io as sio
#
#
# class Pilco65D(object):
#     def __init__(self):
#         self.high_dim = int(65)
#         self.input_dim = int(10)
#         self.eng = matlab.engine.start_matlab()
#         try:
#             minimizer_mat = sio.loadmat(
#                 '/homes/rm4216/Desktop/ImperialCollege/Python/Github_tf_quantile_bo/BayesOpt/tasks/paramsPilco65D/minimizer67D.mat')
#             self.eng.addpath(
#                 '/homes/rm4216/Desktop/ImperialCollege/Python/GitHub/QGP-BO/Experiments/Pilco 65D/pilcoV0.9')
#             self.file_path = 'homes/rm4216/Desktop/ImperialCollege/Python/GitHub/QGP-BO/Experiments/Pilco 65D/pilcoV0.9/'
#         except TypeError:
#             minimizer_mat = sio.loadmat(
#                 '/home/rm4216/Desktop/ImperialCollege/Python/Github-tf_quantile_bo/tasks/paramsPilco65D/minimizer67D.mat')
#             self.eng.addpath(
#                 '/home/rm4216/Desktop/ImperialCollege/Python/GitHub/QGP-BO/Experiments/Pilco 65D/pilcoV0.9')
#             self.file_path = 'home/rm4216/Desktop/ImperialCollege/Python/GitHub/QGP-BO/Experiments/Pilco 65D/pilcoV0.9/'
#         self.bounds = \
#             {
#                 'hyp_bounds': [0.5, 4.],
#                 'input_bounds':
#                     {
#                         '0': [-8., 8.],         # [-5., 5.]
#                         '1': [-15., 15.],       # [-10., 10.]
#                         '2': [-11., 11.],
#                         '34': [-np.pi, np.pi],  # [-np.pi, np.pi],      # why not
#                     },
#                 'target_bounds': [-22., 22.],
#             }
#         self.size_decomposition = int(10)
#         self.num_dim_input = int(5)
#         self.Dinput = int(50)
#         self.decomposition_inputs = [np.arange(start=i, stop=self.Dinput, step=self.num_dim_input) for i in range(self.num_dim_input-2)] + \
#                                     [np.concatenate([np.arange(start=i, stop=self.Dinput, step=self.num_dim_input)+self.num_dim_input-2 for i in range(2)], axis=0)]
#             # [np.arange(start=i*self.size_decomposition, stop=(i+1)*self.size_decomposition) for i in range(self.num_dim_input-2)] + \
#             #                         [np.arange(start=self.size_decomposition*(self.num_dim_input-2), stop=self.size_decomposition*self.num_dim_input)]  # structure: 0, 1, 2, 34
#         # minimizer_mat = sio.loadmat('/homes/rm4216/Desktop/ImperialCollege/Python/Github_tf_quantile_bo/BayesOpt/tasks/paramsPilco65D/minimizer67D.mat')
#         minimizer = minimizer_mat['pilco67D_minimizer'].transpose()    # this has 67 parameters (including fixed)
#         # minimizer_within_bounds = self.clip_values(np.copy(minimizer))
#         self.minimizer = np.copy(minimizer)     # minimizer_within_bounds
#         self.eng.addpath(r'/' + self.file_path + 'scenarios/cartPole')
#         self.eng.addpath(r'/' + self.file_path + 'util')
#         self.eng.addpath(r'/' + self.file_path + 'base')
#         self.eng.addpath(r'/' + self.file_path + 'control')
#         self.eng.addpath(r'/' + self.file_path + 'gp')
#         self.eng.addpath(r'/' + self.file_path + 'loss')
#         self.scale = 0.01
#         self.optimization_indices = np.concatenate([np.arange(start=int(0), stop=int(5), dtype=int),
#                                                np.arange(start=int(7), stop=int(57), dtype=int),
#                                                np.arange(start=int(57), stop=int(67), dtype=int)], axis=0)
#         self.fmin = self.f(self.minimizer, fulldim=True, noisy=False)   # , truescale=False)
#
#     def true_scale(self, x):
#         '''
#         Assuming input x has coordinates between zero and 1
#         :param x: N x 65, add 2 fixed variables for hyp
#         :return: input in the true range
#         '''
#         if x.shape[1] == 67: raise ValueError('Unexpected shape of input for true_scale scale')  # inputs are only allowed to be 65-D
#         bounds_hyp       = self.bounds['hyp_bounds']
#         bounds_inputs_0  = self.bounds['input_bounds']['0']
#         bounds_inputs_1  = self.bounds['input_bounds']['1']
#         bounds_inputs_2  = self.bounds['input_bounds']['2']
#         bounds_inputs_34 = self.bounds['input_bounds']['34']      # previously agreed between [-np.pi, np.pi], clipping the minimizer within these values returns worse value (39.9)
#         bounds_targets   = self.bounds['target_bounds']
#         hyp_rescaled = np.copy(x[:, 0:5]) * (bounds_hyp[1]-bounds_hyp[0]) + bounds_hyp[0]
#         inputs = np.copy(x[:, 5:55])
#         input0_indices = np.ravel(self.decomposition_inputs[0])     # [0 + i*dim_inputs_pilco for i in range(num_local_components)]
#         input1_indices = np.ravel(self.decomposition_inputs[1])     # [1 + i*dim_inputs_pilco for i in range(num_local_components)]
#         input2_indices = np.ravel(self.decomposition_inputs[2])     # [2 + i*dim_inputs_pilco for i in range(num_local_components)]
#         input34_indices = np.ravel(self.decomposition_inputs[3])    # [3 + i*dim_inputs_pilco for i in range(num_local_components)] + [4 + i*dim_inputs_pilco for i in range(num_local_components)]
#         inputs[:, input0_indices] =  inputs[:, input0_indices]  * (bounds_inputs_0[1]  - bounds_inputs_0[0])  + bounds_inputs_0[0]
#         inputs[:, input1_indices] =  inputs[:, input1_indices]  * (bounds_inputs_1[1]  - bounds_inputs_1[0])  + bounds_inputs_1[0]
#         inputs[:, input2_indices] =  inputs[:, input2_indices]  * (bounds_inputs_2[1]  - bounds_inputs_2[0])  + bounds_inputs_2[0]
#         inputs[:, input34_indices] = inputs[:, input34_indices] * (bounds_inputs_34[1] - bounds_inputs_34[0]) + bounds_inputs_34[0]
#         targets_rescaled = np.copy(x[:, 55:65]) * (bounds_targets[1]-bounds_targets[0]) + bounds_targets[0]
#         fixed_hyps = np.tile(np.array([0., -4.605170185988091]), reps=[x.shape[0], 1])
#         return np.column_stack([hyp_rescaled, fixed_hyps, inputs, targets_rescaled])
#
#     def clip_values(self, x):
#         if x.shape[1] != 67: raise ValueError('clip_values function expects 67-D inputs')
#         hyp_bounds = self.bounds['hyp_bounds']
#         x_hyp = np.clip(x[:, 0:5], a_min=hyp_bounds[0], a_max=hyp_bounds[1])
#         target_bounds = self.bounds['target_bounds']
#         x_targets = np.clip(x[:, 57:67], a_min=target_bounds[0], a_max=target_bounds[1])
#         x_inputs = np.copy(x[:, 7:57])
#         indices_inputs0 = np.ravel(self.decomposition_inputs[0])
#         indices_inputs1 = np.ravel(self.decomposition_inputs[1])
#         indices_inputs2 = np.ravel(self.decomposition_inputs[2])
#         indices_inputs34 = np.ravel(self.decomposition_inputs[3])
#         bounds_input0 = self.bounds['input_bounds']['0']
#         x_inputs[:, indices_inputs0]   = np.clip(x_inputs[:, indices_inputs0], a_min=bounds_input0[0], a_max=bounds_input0[1])
#         bounds_input1 = self.bounds['input_bounds']['1']
#         x_inputs[:, indices_inputs1]   = np.clip(x_inputs[:, indices_inputs1], a_min=bounds_input1[0], a_max=bounds_input1[1])
#         bounds_input2 = self.bounds['input_bounds']['2']
#         x_inputs[:, indices_inputs2]   = np.clip(x_inputs[:, indices_inputs2], a_min=bounds_input2[0], a_max=bounds_input2[1])
#         bounds_input34 = self.bounds['input_bounds']['34']
#         x_inputs[:, indices_inputs34] = np.clip(x_inputs[:, indices_inputs34], a_min=bounds_input34[0], a_max=bounds_input34[1])
#         return np.concatenate([x_hyp, np.tile(np.array([0., -4.605170185988091]), reps=[x.shape[0], 1]), x_inputs, x_targets], axis=1)  # working
#
#     def generate_initializations(self, episode_number, num_starts=20):
#         np.random.seed(1552 + episode_number)   # repeatable random inputs
#         num_episodes = int(1)   # fixed to 1
#         try:
#             _inputs_starts_mat = sio.loadmat('/homes/rm4216/Desktop/ImperialCollege/Python/Github_tf_quantile_bo/BayesOpt/tasks/paramsPilco65D/inputs_pilco65_200x50.mat')
#         except TypeError:
#             _inputs_starts_mat = sio.loadmat(
#                 '/home/rm4216/Desktop/ImperialCollege/Python/Github-tf_quantile_bo/tasks/paramsPilco65D/inputs_pilco65_200x50.mat')
#         _inputs_starts_load = np.copy(_inputs_starts_mat['mat_inputs_all_seldim_shape200x50'][episode_number*num_starts:(episode_number+1)*num_starts, :])   # single episode selection, not scaled in [0, 1]
#         _inputs_starts_load_clip = np.concatenate([np.zeros([num_episodes * num_starts, 7]),
#                                                    _inputs_starts_load,
#                                                    np.zeros([num_episodes * num_starts, 10])], axis=1)
#         _inputs_starts_load_clipped = self.clip_values(_inputs_starts_load_clip)                                        # clip within bounds in 67D (but effectively ONLY inputs!)
#         _inputs_starts_load_clipped_scale = np.copy(self.uniform_scale(_inputs_starts_load_clipped)[:, 7:57])           # scale in [0, 1]
#         _inputs_starts = np.reshape(_inputs_starts_load_clipped_scale, [num_episodes, num_starts, self.num_dim_input*self.size_decomposition])  # rehsape with shape: [num_episodes, num_starts, (num_dim_input*size_decomposition)=50]
#         # add scaled hyps and targets
#         num_dim_hyp = int(5)
#         _hyp_starts = np.random.uniform(low=0., high=1., size=num_episodes*num_starts*num_dim_hyp).reshape([num_episodes, num_starts, num_dim_hyp])
#         num_dim_targets = int(10)
#         _target_starts = np.random.uniform(low=0., high=1., size=num_episodes*num_starts*num_dim_targets).reshape([num_episodes, num_starts, num_dim_targets])
#         return np.copy(np.concatenate([_hyp_starts, _inputs_starts, _target_starts], axis=2))   # 65D output
#
#     def uniform_scale(self, x):
#         if x.shape[1] == 67: correction_index = int(2)  # inputs may be 65-D or 67-D according to whether it is the minimizer or not
#         elif x.shape[1] == 65: correction_index = int(0)
#         else: raise ValueError('Unexpected shape of input for uniform_scale')
#         x_hyp = np.copy(x[:, 0:5+correction_index])
#         x_inputs = np.copy(x[:, 5+correction_index:55+correction_index])
#         x_targets = np.copy(x[:, 55+correction_index:65+correction_index])
#         bounds_hyp       = self.bounds['hyp_bounds']
#         bounds_inputs0  = self.bounds['input_bounds']['0']
#         bounds_inputs1  = self.bounds['input_bounds']['1']
#         bounds_inputs2  = self.bounds['input_bounds']['2']
#         bounds_inputs34 = self.bounds['input_bounds']['34']      # previously agreed between [-np.pi, np.pi], clipping the minimizer within these values returns worse value (39.9)
#         bounds_targets   = self.bounds['target_bounds']
#         hyp_uniform = (x_hyp - bounds_hyp[0])/(bounds_hyp[1] - bounds_hyp[0])
#         indices_inputs0 = np.ravel(self.decomposition_inputs[0])
#         x_inputs[:, indices_inputs0] = (x_inputs[:, indices_inputs0] - bounds_inputs0[0])/(bounds_inputs0[1] - bounds_inputs0[0])
#         indices_inputs1 = np.ravel(self.decomposition_inputs[1])
#         x_inputs[:, indices_inputs1] = (x_inputs[:, indices_inputs1] - bounds_inputs1[0])/(bounds_inputs1[1] - bounds_inputs1[0])
#         indices_inputs2 = np.ravel(self.decomposition_inputs[2])
#         x_inputs[:, indices_inputs2] = (x_inputs[:, indices_inputs2] - bounds_inputs2[0])/(bounds_inputs2[1] - bounds_inputs2[0])
#         indices_inputs34 = np.ravel(self.decomposition_inputs[3])
#         x_inputs[:, indices_inputs34] = (x_inputs[:, indices_inputs34] - bounds_inputs34[0])/(bounds_inputs34[1] - bounds_inputs34[0])
#         targets_uniform = (x_targets-bounds_targets[0])/(bounds_targets[1]-bounds_targets[0])
#         return np.concatenate([hyp_uniform, x_inputs, targets_uniform], axis=1)
#
#     def f(self, x, deriv=False, noisy=False, fulldim=False):
#         if len(x.shape) == 1: x = x[None]
#         if fulldim:
#             x = np.copy(x)
#         else:
#             x = self.true_scale(x)
#         # if truescale:
#         #     x = self.true_scale(x)
#         f_list = []
#         df_dx=[]
#         # self.eng.cd(r'/'+self.file_path+'scenarios/cartPole')
#         for x_row in list(x):
#             x_row_mat = matlab.double(x_row[None].tolist())
#             f, df = self.eng.pilco65D(x_row_mat, nargout=2)
#             df_dx.append(np.copy(np.array(df)[:, 0]))
#             f_list.append(f)
#
#         f_x = np.array(f_list)[:, None]
#         if noisy:
#             # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
#             noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
#             y = f_x + noise
#             return y
#         if deriv: return f_x, np.copy(np.stack(df_dx, axis=0)[:, self.optimization_indices])   # Nx1, Nx65
#         return f_x   # Nx1
#
#     def minimize_with_matlab(self, x0, xs_optim=True):
#         if len(x0.shape) == 1: x0 = x0[None]
#         if x0.shape[1] == 65: x0 = self.true_scale(x0)
#         history_opt = []
#         Xopt = []
#         # self.eng.cd(r'/' + self.file_path + 'scenarios/cartPole')
#         for x0_i in list(x0):
#             x0_i_mat = matlab.double(x0_i[None].tolist())
#             Xopt_i, fs_i = self.eng.minimize_pilco65_BFGS(x0_i_mat, nargout=2)
#             # Xopt_i, fs_i = self.eng.minimizePilco65(x0_i_mat, nargout=2)
#             Xopt.append(Xopt_i)
#             history_opt.append(fs_i)
#         if xs_optim: return np.stack(Xopt, axis=0), np.concatenate(history_opt, axis=0).transpose()
#         return np.concatenate(history_opt, axis=1)
#
# # pilco_controller = Pilco65D()
# # np.random.seed(1552)
# # # X = np.random.uniform(low=0., high=1., size=65*5).reshape([5, 65])
# # Xgenerate = pilco_controller.generate_initializations(episode_number=0, num_starts=1)   # episode number from 0 to 9
# # f_gen = []
# # for X_epi_i in list(Xgenerate):
# #     f_gen.append(pilco_controller.f(X_epi_i, truescale=True))
# # f_gen_all = np.concatenate(f_gen, axis=0)
# # # f_X = pilco_controller.f(X, truescale=True)
# # # f_min = pilco_controller.f(X[f_X.argmin(), :][None], truescale=False)
# # # f_min100 = [pilco_controller.f(X[f_X.argmin(), :][None]) for i in range(100)]
# # # noise_stddev_from_min = np.std(f_min100)
# # f_min100_from_minimizer = [pilco_controller.f(pilco_controller.minimizer, truescale=False) for i in range(5)]
# # # f_min100_from_minimizer = [pilco_controller.f(pilco_controller.minimizer, truescale=False) for i in range(100)]
# # f_min100_std = np.std(f_min100_from_minimizer)
# # aaa = 5

class ElectronSphere9np(object):
    def __init__(self):
        self.n_p = int(9)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim

        # if self.n_p == 5
        # self.fmin = 6.4746914
        # self.minimizer = np.array([[0.39655017, 0.52078511, 0.140236, 0.67323942, 0.34780392, 0.02200443, 0.65236761,
        #                             0.65967775, 0.89661884, 0.47885008]])
        self.fmin = 25.759987
        self.minimizer = np.array([[0.5965926, 0.53901935, 0.7879406, 0.49276188, 0.20936081, 0.33062902, 0.21399158,
                                    0.71518286, 0.99539223, 0.60927018, 0.95319699, 0.23248716, 0.70285202, 0.90039541,
                                    0.53554447, 0.1667014, 0.40301371, 0.53355599]])
        self.scale = 0.01

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''
        # # 2D vector
        # if len(alpha.shape) == 1:
        #     alpha = alpha[None]
        # # convert to cartesian
        # alpha_reshape = np.reshape(alpha, newshape=[np.shape(alpha)[0], self.n_p, int(2)])  # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        assert np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
            alpha_reshape >= 0.)
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])  # N x (\alpha_i, \beta_i), for i=1,...,self.n_p
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)     # \alpha \in [0, 1] -> \theta \in [0, 2\pi]
        phi = np.copy(x_reshape[:, :, 1] * np.pi)           # \beta  \in [0, 1] -> \phi   \in [0, \pi]
        spherical = np.stack([theta, phi], axis=-1)                             # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        x, y, z = self.spherical_to_cartesian(spherical)

        # x_all = x.ravel()
        # y_all = y.ravel()
        # z_all = z.ravel()
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_all, y_all, z_all, zdir='z', s=20, c=None, depthshade=True)
        # plt.show()

        x_Mat = x[:, :, None] - x[:, None, :]
        # for k in range(x.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert x[k, i] - x[k, j] == x_Mat[k, i, j]
        y_Mat = y[:, :, None] - y[:, None, :]
        # for k in range(y.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert y[k, i] - y[k, j] == y_Mat[k, i, j]
        z_Mat = z[:, :, None] - z[:, None, :]
        # for k in range(z.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert z[k, i] - z[k, j] == z_Mat[k, i, j]

        x_Mat2 = x_Mat ** 2.0
        # assert np.all(x_Mat2 == x_Mat2.transpose((0, 2, 1)))
        y_Mat2 = y_Mat ** 2.0
        # assert np.all(y_Mat2 == y_Mat2.transpose((0, 2, 1)))
        z_Mat2 = z_Mat ** 2.0
        # assert np.all(z_Mat2 == z_Mat2.transpose((0, 2, 1)))

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)
        # assert np.all(Mat2 == Mat2.transpose((0, 2, 1)))

        # trace_term = np.trace(Mat2, axis1=1, axis2=2)[:, None]
        # assert np.all(trace_term == 0.0)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf:
            f_x[f_x[:, 0] == np.inf, :] = 1e09
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1



class ElectronSphere6np(object):
    def __init__(self):
        self.n_p = int(6)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim

        # if self.n_p == 5
        # self.fmin = 6.4746914
        # self.minimizer = np.array([[0.39655017, 0.52078511, 0.140236, 0.67323942, 0.34780392, 0.02200443, 0.65236761,
        #                             0.65967775, 0.89661884, 0.47885008]])
        self.fmin = 9.985281
        self.minimizer = np.array([[0.48682843, 0.78674212, 0.57885328, 0.31341441, 0.29749929, 0.40868617, 0.07889095,
                                    0.68647393, 0.98678081, 0.2134249, 0.79744743, 0.59130902]])
        self.scale = 0.01

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''
        # # 2D vector
        # if len(alpha.shape) == 1:
        #     alpha = alpha[None]
        # # convert to cartesian
        # alpha_reshape = np.reshape(alpha, newshape=[np.shape(alpha)[0], self.n_p, int(2)])  # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        if np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
            alpha_reshape >= 0.):
            print('Input ElectronSphere6np as expected')
        else:
            print(alpha_reshape)
        # assert np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
        #     alpha_reshape >= 0.)
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])  # N x (\alpha_i, \beta_i), for i=1,...,self.n_p
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)     # \alpha \in [0, 1] -> \theta \in [0, 2\pi]
        phi = np.copy(x_reshape[:, :, 1] * np.pi)           # \beta  \in [0, 1] -> \phi   \in [0, \pi]
        spherical = np.stack([theta, phi], axis=-1)                             # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        x, y, z = self.spherical_to_cartesian(spherical)

        # x_all = x.ravel()
        # y_all = y.ravel()
        # z_all = z.ravel()
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_all, y_all, z_all, zdir='z', s=20, c=None, depthshade=True)
        # plt.show()

        x_Mat = x[:, :, None] - x[:, None, :]
        # for k in range(x.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert x[k, i] - x[k, j] == x_Mat[k, i, j]
        y_Mat = y[:, :, None] - y[:, None, :]
        # for k in range(y.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert y[k, i] - y[k, j] == y_Mat[k, i, j]
        z_Mat = z[:, :, None] - z[:, None, :]
        # for k in range(z.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert z[k, i] - z[k, j] == z_Mat[k, i, j]

        x_Mat2 = x_Mat ** 2.0
        # assert np.all(x_Mat2 == x_Mat2.transpose((0, 2, 1)))
        y_Mat2 = y_Mat ** 2.0
        # assert np.all(y_Mat2 == y_Mat2.transpose((0, 2, 1)))
        z_Mat2 = z_Mat ** 2.0
        # assert np.all(z_Mat2 == z_Mat2.transpose((0, 2, 1)))

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)
        # assert np.all(Mat2 == Mat2.transpose((0, 2, 1)))

        # trace_term = np.trace(Mat2, axis1=1, axis2=2)[:, None]
        # assert np.all(trace_term == 0.0)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf or f_x.max() >= 1e09:
            f_x[np.logical_or(f_x[:, 0] == np.inf, f_x[:, 0] >= 1e09), :] = 1e09
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

class ElectronSphere(object):
    def __init__(self):
        self.n_p = int(25)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim

        self.q, self.q_inv = self.linear_map()

        self.fmin = 2.43812 * 1e02
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''
        # # 2D vector
        # if len(alpha.shape) == 1:
        #     alpha = alpha[None]
        # # convert to cartesian
        # alpha_reshape = np.reshape(alpha, newshape=[np.shape(alpha)[0], self.n_p, int(2)])  # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        assert np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
            alpha_reshape >= 0.)
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])  # N x (\alpha_i, \beta_i), for i=1,...,self.n_p
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)     # \alpha \in [0, 1] -> \theta \in [0, 2\pi]
        phi = np.copy(x_reshape[:, :, 1] * np.pi)           # \beta  \in [0, 1] -> \phi   \in [0, \pi]
        spherical = np.stack([theta, phi], axis=-1)                             # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        x, y, z = self.spherical_to_cartesian(spherical)

        # x_all = x.ravel()
        # y_all = y.ravel()
        # z_all = z.ravel()
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_all, y_all, z_all, zdir='z', s=20, c=None, depthshade=True)
        # plt.show()

        x_Mat = x[:, :, None] - x[:, None, :]
        # for k in range(x.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert x[k, i] - x[k, j] == x_Mat[k, i, j]
        y_Mat = y[:, :, None] - y[:, None, :]
        # for k in range(y.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert y[k, i] - y[k, j] == y_Mat[k, i, j]
        z_Mat = z[:, :, None] - z[:, None, :]
        # for k in range(z.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert z[k, i] - z[k, j] == z_Mat[k, i, j]

        x_Mat2 = x_Mat ** 2.0
        # assert np.all(x_Mat2 == x_Mat2.transpose((0, 2, 1)))
        y_Mat2 = y_Mat ** 2.0
        # assert np.all(y_Mat2 == y_Mat2.transpose((0, 2, 1)))
        z_Mat2 = z_Mat ** 2.0
        # assert np.all(z_Mat2 == z_Mat2.transpose((0, 2, 1)))

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)
        # assert np.all(Mat2 == Mat2.transpose((0, 2, 1)))

        # trace_term = np.trace(Mat2, axis1=1, axis2=2)[:, None]
        # assert np.all(trace_term == 0.0)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf:
            f_x = np.ones_like(f_x) * 1e09
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# spherical_obj = ElectronSphere()
# np.random.seed(0)
# N = int(500)
# X = np.random.uniform(low=0., high=1., size=[N, spherical_obj.high_dim])
# f = spherical_obj.f(X, noisy=False, fulldim=False)



class ElectronSphereRot(object):
    def __init__(self):
        self.n_p = int(25)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim

        self.q, self.q_inv = self.linear_map()

        self.fmin = 2.43812 * 1e02
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''
        # # 2D vector
        # if len(alpha.shape) == 1:
        #     alpha = alpha[None]
        # # convert to cartesian
        # alpha_reshape = np.reshape(alpha, newshape=[np.shape(alpha)[0], self.n_p, int(2)])  # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        # assert np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
        #     alpha_reshape >= 0.)
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        xm1_1 = (x * 2.) - 1.                       # bring to interval [-1, 1]
        x = np.matmul(xm1_1, self.q.transpose())    # rotate [-1, 1] -> [-1, 1]
        x = (np.copy(x) + 1.) / 2.                  # bring to interval [0, 1]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])  # N x (\alpha_i, \beta_i), for i=1,...,self.n_p
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)     # \alpha \in [0, 1] -> \theta \in [0, 2\pi]
        phi = np.copy(x_reshape[:, :, 1] * np.pi)           # \beta  \in [0, 1] -> \phi   \in [0, \pi]
        spherical = np.stack([theta, phi], axis=-1)                             # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        x, y, z = self.spherical_to_cartesian(spherical)

        # x_all = x.ravel()
        # y_all = y.ravel()
        # z_all = z.ravel()
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_all, y_all, z_all, zdir='z', s=20, c=None, depthshade=True)
        # plt.show()

        x_Mat = x[:, :, None] - x[:, None, :]
        # for k in range(x.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert x[k, i] - x[k, j] == x_Mat[k, i, j]
        y_Mat = y[:, :, None] - y[:, None, :]
        # for k in range(y.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert y[k, i] - y[k, j] == y_Mat[k, i, j]
        z_Mat = z[:, :, None] - z[:, None, :]
        # for k in range(z.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert z[k, i] - z[k, j] == z_Mat[k, i, j]

        x_Mat2 = x_Mat ** 2.0
        # assert np.all(x_Mat2 == x_Mat2.transpose((0, 2, 1)))
        y_Mat2 = y_Mat ** 2.0
        # assert np.all(y_Mat2 == y_Mat2.transpose((0, 2, 1)))
        z_Mat2 = z_Mat ** 2.0
        # assert np.all(z_Mat2 == z_Mat2.transpose((0, 2, 1)))

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)
        # assert np.all(Mat2 == Mat2.transpose((0, 2, 1)))

        # trace_term = np.trace(Mat2, axis1=1, axis2=2)[:, None]
        # assert np.all(trace_term == 0.0)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf:
            f_x = np.ones_like(f_x) * 1e09
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1



class ElectronSphereRot100np(object):
    def __init__(self):
        self.n_p = int(100)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim

        self.q, self.q_inv = self.linear_map()

        self.fmin = 4.44841 * 1e03
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''
        # # 2D vector
        # if len(alpha.shape) == 1:
        #     alpha = alpha[None]
        # # convert to cartesian
        # alpha_reshape = np.reshape(alpha, newshape=[np.shape(alpha)[0], self.n_p, int(2)])  # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        # assert np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
        #     alpha_reshape >= 0.)
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        xm1_1 = (x * 2.) - 1.                       # bring to interval [-1, 1]
        x = np.matmul(xm1_1, self.q.transpose())    # rotate [-1, 1] -> [-1, 1]
        x = (np.copy(x) + 1.) / 2.                  # bring to interval [0, 1]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])  # N x (\alpha_i, \beta_i), for i=1,...,self.n_p
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)     # \alpha \in [0, 1] -> \theta \in [0, 2\pi]
        phi = np.copy(x_reshape[:, :, 1] * np.pi)           # \beta  \in [0, 1] -> \phi   \in [0, \pi]
        spherical = np.stack([theta, phi], axis=-1)                             # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        x, y, z = self.spherical_to_cartesian(spherical)

        # x_all = x.ravel()
        # y_all = y.ravel()
        # z_all = z.ravel()
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_all, y_all, z_all, zdir='z', s=20, c=None, depthshade=True)
        # plt.show()

        x_Mat = x[:, :, None] - x[:, None, :]
        # for k in range(x.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert x[k, i] - x[k, j] == x_Mat[k, i, j]
        y_Mat = y[:, :, None] - y[:, None, :]
        # for k in range(y.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert y[k, i] - y[k, j] == y_Mat[k, i, j]
        z_Mat = z[:, :, None] - z[:, None, :]
        # for k in range(z.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert z[k, i] - z[k, j] == z_Mat[k, i, j]

        x_Mat2 = x_Mat ** 2.0
        # assert np.all(x_Mat2 == x_Mat2.transpose((0, 2, 1)))
        y_Mat2 = y_Mat ** 2.0
        # assert np.all(y_Mat2 == y_Mat2.transpose((0, 2, 1)))
        z_Mat2 = z_Mat ** 2.0
        # assert np.all(z_Mat2 == z_Mat2.transpose((0, 2, 1)))

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)
        # assert np.all(Mat2 == Mat2.transpose((0, 2, 1)))

        # trace_term = np.trace(Mat2, axis1=1, axis2=2)[:, None]
        # assert np.all(trace_term == 0.0)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf:
            f_x = np.ones_like(f_x) * 1e09
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# spherical_obj = ElectronSphereRot100np()
# np.random.seed(0)
# N = int(500)
# X = np.random.uniform(low=0., high=1., size=[N, spherical_obj.high_dim])
# f = spherical_obj.f(X, noisy=False, fulldim=False)
# aaa = 5


class ElectronSphereLog(object):
    def __init__(self):
        self.n_p = int(25)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim

        self.fmin = np.log(2.43812 * 1e02)
        self.scale = 0.01

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''
        # # 2D vector
        # if len(alpha.shape) == 1:
        #     alpha = alpha[None]
        # # convert to cartesian
        # alpha_reshape = np.reshape(alpha, newshape=[np.shape(alpha)[0], self.n_p, int(2)])  # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        assert np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
            alpha_reshape >= 0.)
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])  # N x (\alpha_i, \beta_i), for i=1,...,self.n_p
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)     # \alpha \in [0, 1] -> \theta \in [0, 2\pi]
        phi = np.copy(x_reshape[:, :, 1] * np.pi)           # \beta  \in [0, 1] -> \phi   \in [0, \pi]
        spherical = np.stack([theta, phi], axis=-1)                             # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        x, y, z = self.spherical_to_cartesian(spherical)

        # x_all = x.ravel()
        # y_all = y.ravel()
        # z_all = z.ravel()
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_all, y_all, z_all, zdir='z', s=20, c=None, depthshade=True)
        # plt.show()

        x_Mat = x[:, :, None] - x[:, None, :]
        # for k in range(x.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert x[k, i] - x[k, j] == x_Mat[k, i, j]
        y_Mat = y[:, :, None] - y[:, None, :]
        # for k in range(y.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert y[k, i] - y[k, j] == y_Mat[k, i, j]
        z_Mat = z[:, :, None] - z[:, None, :]
        # for k in range(z.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert z[k, i] - z[k, j] == z_Mat[k, i, j]

        x_Mat2 = x_Mat ** 2.0
        # assert np.all(x_Mat2 == x_Mat2.transpose((0, 2, 1)))
        y_Mat2 = y_Mat ** 2.0
        # assert np.all(y_Mat2 == y_Mat2.transpose((0, 2, 1)))
        z_Mat2 = z_Mat ** 2.0
        # assert np.all(z_Mat2 == z_Mat2.transpose((0, 2, 1)))

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)
        # assert np.all(Mat2 == Mat2.transpose((0, 2, 1)))

        # trace_term = np.trace(Mat2, axis1=1, axis2=2)[:, None]
        # assert np.all(trace_term == 0.0)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf:
            f_x = np.ones_like(f_x) * 1e50
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = np.log(f_x) + noise
            return y
        return np.log(f_x)    # Nx1

# spherical_obj = ElectronSphereLog()
# np.random.seed(0)
# N = int(500)
# X = np.random.uniform(low=0., high=1., size=[N, spherical_obj.high_dim])
# f = spherical_obj.f(X, noisy=False, fulldim=False)



class ElectronSphereLog100np(object):
    def __init__(self):
        self.n_p = int(100)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim

        self.fmin = np.log(4.44841 * 1e03)
        self.scale = 0.01

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''
        # # 2D vector
        # if len(alpha.shape) == 1:
        #     alpha = alpha[None]
        # # convert to cartesian
        # alpha_reshape = np.reshape(alpha, newshape=[np.shape(alpha)[0], self.n_p, int(2)])  # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        # assert np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
        #     alpha_reshape >= 0.)
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])  # N x (\alpha_i, \beta_i), for i=1,...,self.n_p
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)     # \alpha \in [0, 1] -> \theta \in [0, 2\pi]
        phi = np.copy(x_reshape[:, :, 1] * np.pi)           # \beta  \in [0, 1] -> \phi   \in [0, \pi]
        spherical = np.stack([theta, phi], axis=-1)                             # N x (\theta_i, \phi_i), for i=1,...,self.n_p
        x, y, z = self.spherical_to_cartesian(spherical)

        # x_all = x.ravel()
        # y_all = y.ravel()
        # z_all = z.ravel()
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_all, y_all, z_all, zdir='z', s=20, c=None, depthshade=True)
        # plt.show()

        x_Mat = x[:, :, None] - x[:, None, :]
        # for k in range(x.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert x[k, i] - x[k, j] == x_Mat[k, i, j]
        y_Mat = y[:, :, None] - y[:, None, :]
        # for k in range(y.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert y[k, i] - y[k, j] == y_Mat[k, i, j]
        z_Mat = z[:, :, None] - z[:, None, :]
        # for k in range(z.shape[0]):
        #     for i in range(self.n_p):
        #         for j in range(self.n_p):
        #             assert z[k, i] - z[k, j] == z_Mat[k, i, j]

        x_Mat2 = x_Mat ** 2.0
        # assert np.all(x_Mat2 == x_Mat2.transpose((0, 2, 1)))
        y_Mat2 = y_Mat ** 2.0
        # assert np.all(y_Mat2 == y_Mat2.transpose((0, 2, 1)))
        z_Mat2 = z_Mat ** 2.0
        # assert np.all(z_Mat2 == z_Mat2.transpose((0, 2, 1)))

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)
        # assert np.all(Mat2 == Mat2.transpose((0, 2, 1)))

        # trace_term = np.trace(Mat2, axis1=1, axis2=2)[:, None]
        # assert np.all(trace_term == 0.0)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf:
            f_x = np.ones_like(f_x) * 1e50
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = np.log(f_x) + noise
            return y
        return np.log(f_x)    # Nx1

# spherical_obj = ElectronSphereLog100np()
# np.random.seed(0)
# N = int(500)
# X = np.random.uniform(low=0., high=1., size=[N, spherical_obj.high_dim])
# f = spherical_obj.f(X, noisy=False, fulldim=False)
# aaa = 5



class SwissRoll1D(object):
    def __init__(self):
        self.high_dim = int(2)
        self.input_dim = self.high_dim

        self.X, self.Y = self.generateSR()

        self.x_bounds = [-11.1, 14.2]

        self.minimizer = np.copy(self.X[-1, :])[None]
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=True)
        assert self.fmin == self.Y[-1, :]
        self.scale = 0.01

    def generateSR(self):
        dataset = np.linspace(start=0., stop=1., num=5000)[:, None]
        data_z = (dataset * 2 + 1) * 3 * np.pi / 2.
        data_x = np.multiply(data_z, np.cos(data_z))
        data_y = np.multiply(data_z, np.sin(data_z))
        swiss_roll = np.stack([data_x, data_y], axis=0).transpose()
        function_SR = - dataset * 5.
        return swiss_roll[0, :, :], function_SR

    def square_dists(self, xC, X2):
        xCs = np.sum(np.square(xC), axis=1, keepdims=True)          # M x 1
        Xs = np.sum(np.square(X2), axis=1, keepdims=True)           # N x 1
        xXT = np.matmul(xC, np.transpose(X2))                       # M x N
        KL = (xCs + np.transpose(Xs) - 2. * xXT) / 2.               # M x N
        return KL

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if not fulldim:
            x = x * (self.x_bounds[1] - self.x_bounds[0]) + self.x_bounds[0]
        pw_dists2 = self.square_dists(x, self.X)
        indices_min = np.argmin(pw_dists2, axis=1)

        indices_m = np.arange(x.shape[0])
        f_x = np.multiply(self.Y[indices_min, :], np.exp(-pw_dists2[indices_m, indices_min])[:, None])


        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1




class HartmannNN6D(object):
    def __init__(self):
        self.high_dim = int(60)
        self.input_dim = int(6)
        self.A = np.array([[10., 3., 17., 3.50, 1.7, 8],
                           [0.05, 10., 17., 0.1, 8., 14.],
                           [3., 3.5, 1.7, 10., 17., 8.],
                           [17., 8., 0.05, 10., 0.1, 14.]])
        self.P = 1e-04 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                                   [2329., 4135., 8307., 3736., 1004., 9991.],
                                   [2348., 1451., 3522., 2883., 3047., 6650.],
                                   [4047., 8828., 8732., 5743., 1091., 381.]])
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])

        self.q, self.q_inv = self.linear_map()
        self.minimizer6 = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[
            None]
        self.fmin6 = self.f(self.minimizer6, noisy=False, fulldim=True)
        self.minimizer = (np.matmul(self.sigmoid_inv(self.minimizer6), self.q_inv.transpose()) + 1.) / 2.         # optimizer in 60-dimensional space in [0,1]
        # self.minimizer = (np.matmul(self.minimizer6 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.         # optimizer in 60-dimensional space in [0,1]
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=False)
        # self.minimizer = np.zeros(shape=[1, self.high_dim], dtype=np.float64)
        # self.minimizer[:, self.relevant_dims] = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[None]
        # self.fmin = self.f(self.minimizer)  # -3.32237
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        # np.save('HartmannLinear6D' + str(self.high_dim) + '_' + str(self.input_dim) + '.npy', Mat)
        # Mat = loadMat('HartmannLinear6D' + str(self.high_dim) + '_' + str(self.input_dim) + '.npy')
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def sigmoid(self, x):
        return np.divide(1., 1. + np.exp(-x))

    def sigmoid_inv(self, z):
        return np.log(np.divide(z, 1. - z))

    def f(self, x, noisy=False, fulldim=False):
        '''
        x is assumed 60-dimensional between [0, 1], shape Nx60
        :param x: Nx60
        :return: f_x(x) Nx1
        '''
        if len(x.shape) == 1:
            x = x[None]

        if fulldim:
            x = x
        else:
            x_m1_1 = x * 2. - 1.                        # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())   # bring to 6-dimensional mostly in [-1, 1]
            x = self.sigmoid(x)                         # bring to [0, 1] with Sigmoid activation function

        # From here, x is assumed 6-dimensional and between [0, 1]
        val_i = []
        for x_i in list(x):     # x_i 1x6
            exponents = np.sum(np.multiply(np.square(self.P - x_i), self.A), axis=1)
            val_i.append(-np.sum(np.multiply(np.exp(-exponents), self.alpha), axis=0))
        f_x = np.vstack(val_i)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# hartNN = HartmannNN6D()
# np.random.seed(0)
# X = np.random.uniform(low=0., high=1., size=[1000000, hartNN.high_dim])
# sig = hartNN.sigmoid(X)
# X_inv = hartNN.sigmoid_inv(sig)
# err = np.max(np.abs(X - X_inv))
#
# f_x = hartNN.f(X, noisy=False, fulldim=False)
# aaa = 5.


class ProductSinesNN10D(object):
    def __init__(self):
        self.high_dim = int(60)
        self.input_dim = int(10)

        self.q, self.q_inv = self.linear_map()

        # self.minimizer10 = np.ones(shape=[1, self.input_dim]) * ((np.pi * (3. / 2.)) - np.pi ) / (np.pi)
        self.minimizer10 = np.ones(shape=[1, self.input_dim]) * (np.pi * (3. / 2.)) / (2. * np.pi)
        self.fmin10 = self.f(self.minimizer10, noisy=False, fulldim=True)
        self.minimizer = (np.matmul(self.sigmoid_inv(self.minimizer10), self.q_inv.transpose()) + 1.) / 2.         # optimizer in 60-dimensional space in [0,1]
        # self.minimizer = (np.matmul(self.minimizer10 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=False)
        self.scale = 0.01

    def linear_map(self):
        # np.random.seed(44)
        np.random.seed(30)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        # np.save('ProductSinesLinear10D_' + str(self.high_dim) + '_' + str(self.input_dim) + '.npy', Mat)
        # Mat = loadMat('ProductSinesLinear10D_' + str(self.high_dim) + '_' + str(self.input_dim) + '.npy')
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        # q_norm = np.linalg.norm(q, ord=None, axis=1, keepdims=True)
        return q, np.linalg.pinv(q)

    def sigmoid(self, x):
        return np.divide(1., 1. + np.exp(-x))

    def sigmoid_inv(self, z):
        return np.log(np.divide(z, 1. - z))

    def f(self, x, noisy=False, fulldim=False):
        if len(x.shape) == 1:
            x = x[None]

        if fulldim:
            x_resc = x * (2. * np.pi)                       # [0, 1] -> [0, 2pi]                                                    Not [pi, 2pi]
        else:
            x_m1_1 = x * 2. - 1.                            # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())       # bring to 10-dimensional mostly [-1, 1]
            x01 = self.sigmoid(x)                           # bring to [0, 1] with Sigmoid activation function
            x_resc = np.copy(x01) * (2. * np.pi)            # (ideally, only true for the optimum) [0, 1] -> [0, 2pi]              Not [pi, 2pi]
        f_x = np.sin(x_resc[:, 0])[:, None] * np.prod(np.sin(x_resc), axis=1)[:, None] * 10.           # rescale by factor
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # + np.random.normal(loc=0., scale=0.1, size=1)

# psNN = ProductSinesNN10D()
# np.random.seed(0)
# X = np.random.uniform(low=0., high=1., size=[1000000, psNN.high_dim])
# sig = psNN.sigmoid(X)
# X_inv = psNN.sigmoid_inv(sig)
# err = np.max(np.abs(X - X_inv))
#
# f_x = psNN.f(X, noisy=False, fulldim=False)
# aaa = 5.


class RosenbrockNN10D(object):
    def __init__(self):
        self.high_dim = int(60)
        self.input_dim = int(10)

        self.lb = -2.048
        self.ub = 2.048

        self.q, self.q_inv = self.linear_map()
        self.minimizer10 = (np.ones(shape=[1, self.input_dim]) - self.lb) / (self.ub - self.lb)
        self.fmin10 = self.f(self.minimizer10, noisy=False, fulldim=True)
        self.minimizer = (np.matmul(self.sigmoid_inv(self.minimizer10),
                                    self.q_inv.transpose()) + 1.) / 2.  # optimizer in 60-dimensional space in [0,1]
        # self.minimizer = (np.matmul(self.minimizer10 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.
        self.fmin = self.f(self.minimizer, fulldim=False, noisy=False)      # fmin = 0
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(20)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def sigmoid(self, x):
        return np.divide(1., 1. + np.exp(-x))

    def sigmoid_inv(self, z):
        return np.log(np.divide(z, 1. - z))

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is Nx10 with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x_resc = x * (self.ub - self.lb) + self.lb                                  # Nx10
        else:
            x_m1_1 = x * 2. - 1.                            # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())       # bring to 10-dimensional # [-1, 1] -> [-1, 1]
            x = self.sigmoid(x)                             # bring to [0, 1] with Sigmoid activation function
            x_resc = np.copy(x) * (self.ub - self.lb) + self.lb  # Nx10
        xi = x_resc[:, :-1].copy()
        xip1 = x_resc[:, 1:].copy()
        addend = 100. * (xip1 - xi ** 2.) ** 2. + (xi - 1.) ** 2.
        f_x = np.sum(addend, axis=1, keepdims=True)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# RosNN = RosenbrockNN10D()
# np.random.seed(0)
# X = np.random.uniform(low=0., high=1., size=[1000000, RosNN.high_dim])
# sig = RosNN.sigmoid(X)
# X_inv = RosNN.sigmoid_inv(sig)
# err = np.max(np.abs(X - X_inv))
#
# f_x = RosNN.f(X, noisy=False, fulldim=False)
# aaa = 5.


class MichalewiczNN10D(object):
    def __init__(self, m=0.5):
        self.high_dim = int(60)
        self.input_dim = int(10)
        self.m = m

        self.q, self.q_inv = self.linear_map()
        self.minimizer10 = np.array([[1.9756, 1.5708, 1.3227, 1.1611, 1.0464, 0.9598, 0.8916,
                                                           1.7539, 1.6548, 1.5708]]) / np.pi
        # self.minimizer10 = np.array([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796,
        #                                                    1.454414, 1.756087, 1.655717, 1.570796]])/np.pi     # m=10
        self.fmin10 = self.f(self.minimizer10, noisy=False, fulldim=True)
        self.minimizer = (np.matmul(self.sigmoid_inv(self.minimizer10),
                                    self.q_inv.transpose()) + 1.) / 2.  # optimizer in 60-dimensional space in [0,1]
        # self.minimizer = (np.matmul(self.minimizer10 * 2. - 1., self.q_inv.transpose()) + 1.) / 2.
        self.fmin = self.f(self.minimizer, noisy=False, fulldim=False)  # m=10., fmin=-9.66015; m=0.5, fmin=-9.1415
        self.scale = 0.01

    def linear_map(self):
        np.random.seed(44)
        Mat = np.random.normal(loc=0., scale=1., size=[self.input_dim, self.high_dim])
        q, r = np.linalg.qr(np.transpose(Mat), mode='reduced')
        q = np.transpose(q)
        return q, np.linalg.pinv(q)

    def sigmoid(self, x):
        return np.divide(1., 1. + np.exp(-x))

    def sigmoid_inv(self, z):
        return np.log(np.divide(z, 1. - z))

    def f(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is Nx10 with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]
        if fulldim:
            x_resc = x * np.pi
        else:
            x_m1_1 = x * 2. - 1.                            # bring to interval [-1, 1]
            x = np.matmul(x_m1_1, self.q.transpose())       # bring to 10-dimensional # [-1, 1] -> [-1, 1]
            x = self.sigmoid(x)                             # bring to [0, 1] with Sigmoid activation function
            x_resc = np.copy(x) * np.pi                     # Nx10
        i_s = np.arange(self.input_dim)[None] + 1
        fact1 = np.sin(np.multiply(i_s, np.square(x_resc))/np.pi) ** (2 * self.m)
        factr2 = np.sin(x_resc)
        f_x = - np.sum(np.multiply(fact1, factr2), axis=1, keepdims=True)
        if noisy:
            # noise = np.random.normal(loc=0., scale=self.scale, size=f_x.shape[0])[:, None]
            noise = np.array([random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1

# MichNN = MichalewiczNN10D()
# np.random.seed(0)
# X = np.random.uniform(low=0., high=1., size=[1000000, MichNN.high_dim])
# sig = MichNN.sigmoid(X)
# X_inv = MichNN.sigmoid_inv(sig)
# err = np.max(np.abs(X - X_inv))
#
# f_x = MichNN.f(X, noisy=False, fulldim=False)
# aaa = 5.