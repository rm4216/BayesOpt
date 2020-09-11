import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x, noisy=False, fulldim=False):
    relevant_dims = np.array([16, 21, 31, 38, 45, 78, 84, 85, 91, 95], dtype=int)
    scale = 0.01
    if len(x.shape) == 1:
        x = x[None]
    if fulldim:
        x_resc = x * (np.pi) + np.pi/2.
    else:
        x_resc = np.copy(x[:, relevant_dims]) * (2. * np.pi)  # [0, 1] -> [0, 2pi]
    f_x = np.sin(x_resc[:, 0])[:, None] * np.prod(np.sin(x_resc), axis=1)[:, None] * 10  # rescale by factor
    if noisy:
        noise = np.random.normal(loc=0., scale=scale, size=f_x.shape[0])[:, None]
        y = f_x + noise
        return y
    return f_x

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(start=0., stop=1., num=100)[:, None]
X1, X2 = np.meshgrid(x, x)
X = np.column_stack([np.reshape(X1, [-1, 1]), np.reshape(X2, [-1, 1])])
fX = f(X, noisy=False, fulldim=True)

ax.plot_surface(X1, X2, np.reshape(fX, X1.shape))
plt.show()