# import numpy as np
# from tfbo.utils.import_modules import import_attr
from autograd import grad
import autograd.numpy as np
from autograd.numpy import sin


# name_task = 'Michalewicz10D'
# task_attr = import_attr('datasets/tasks/all_tasks', attribute=name_task)
# objective = task_attr()
# func = lambda x: objective.f(x, noisy=False, fulldim=True)


# def Michalewicz10D(x, noisy=False, fulldim=False):
#     '''
#     Assuming the input x is Nx10 with values in the range [0, 1]
#     :param x:
#     :return:
#     '''
#     m = 0.5
#     scale = 0.01
#     input_dim = int(10)
#     relevant_dims = np.array([16, 21, 31, 38, 45, 78, 84, 85, 91, 95], dtype=int)
#     if len(x.shape) == 1:
#         x = x[None]
#     if fulldim:
#         x_resc = x * np.pi
#     else:
#         x_resc = np.copy(x[:, relevant_dims]) * np.pi  # Nx10
#     i_s = np.arange(input_dim)[None] + 1
#     fact1 = sin(np.multiply(i_s, np.square(x_resc)) / np.pi) ** (2 * m)
#     factr2 = sin(x_resc)
#     f_x = - np.sum(np.multiply(fact1, factr2), axis=1, keepdims=True)
#     if noisy:
#         noise = np.random.normal(loc=0., scale=scale, size=f_x.shape[0])[:, None]
#         y = f_x + noise
#         return y
#     return f_x
# func = lambda x: Michalewicz10D(x, noisy=False, fulldim=True)
# func_grad = grad(func)
# minimizer = np.array([[1.9756, 1.5708, 1.3227, 1.1611, 1.0464, 0.9598, 0.8916, 1.7539, 1.6548, 1.5708]]) / np.pi
# g0 = func_grad(minimizer)
#
# np.random.seed(123)
# num = int(5e06)
# shape_x = [num, int(10)]
# x = np.random.uniform(low=0., high=1., size=np.prod(shape_x)).reshape(shape_x)
# grads = []
# for x_i, i in zip(list(x), list(range(num))):
#     print(i)
#     grad_i = func_grad(x_i[None])
#     grads.append(grad_i)


def ProductSines10D(x, noisy=False, fulldim=False):
    relevant_dims = np.array([16, 21, 31, 38, 45, 78, 84, 85, 91, 95], dtype=int)
    scale = 0.01
    if len(x.shape) == 1:
        x = x[None]
    if fulldim:
        x_resc = x * (2. * np.pi)
    else:
        x_resc = np.copy(x[:, relevant_dims]) * (2. * np.pi)  # [0, 1] -> [0, 2pi]
    f_x = np.sin(x_resc[:, 0])[:, None] * np.prod(np.sin(x_resc), axis=1)[:, None] * 10  # rescale by factor
    if noisy:
        noise = np.random.normal(loc=0., scale=scale, size=f_x.shape[0])[:, None]
        y = f_x + noise
        return y
    return f_x
func = lambda x: ProductSines10D(x, noisy=False, fulldim=True)
func_grad = grad(func)
minimizer = np.ones(shape=[1, int(100)], dtype=np.float64) * (np.pi * (3. / 2.)) / (2*np.pi)
g0 = func_grad(minimizer)

np.random.seed(123)
num = int(5e06)
shape_x = [num, int(10)]
x = np.random.uniform(low=0., high=1., size=np.prod(shape_x)).reshape(shape_x)
grads = []
for x_i, i in zip(list(x), list(range(num))):
    print(i)
    grad_i = func_grad(x_i[None])
    grads.append(grad_i)



grads_all = np.concatenate(grads, axis=0)
# f_sample = objective.f(x, noisy=False, fulldim=True)
# f_min_sample = f_sample.min()