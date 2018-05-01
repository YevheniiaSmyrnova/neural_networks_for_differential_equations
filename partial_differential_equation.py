"""
Solution of PDE
Error: 0.18696873538824116
"""
import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import cm


nx = 10
ny = 10

dx = 1. / nx
dy = 1. / ny

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)


def analytic_solution(x):
    """
    Analytical solution of current problem
    :param x:
    :return:
    """
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
           np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))


# Draw analytical solution
surface = np.zeros((ny, nx))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 2)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()


def f(x):
    """
    This is f() function on the right
    :param x:
    :return:
    """
    return 0.


def sigmoid(x):
    """
    Sigmoid function
    :param x:
    :return:
    """
    return 1. / (1. + np.exp(-x))


def neural_network(W, x):
    """
    Neural network
    :param W:
    :param x:
    :return:
    """
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])


def A(x):
    """
    Function A for trial solution
    :param x:
    :return:
    """
    return x[1] * np.sin(np.pi * x[0])


def psy_trial(x, net_out):
    """
    Trial solution
    :param xi:
    :param net_out:
    :return:
    """
    return A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out


def loss_function(W, x, y):
    """
    Loss function
    :param W:
    :param x:
    :param y:
    :return:
    """
    loss_sum = 0.
    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])
            net_out = neural_network(W, input_point)[0]
            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)
            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]
            func = f(input_point)
            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func) ** 2
            loss_sum += err_sqr
    return loss_sum


# Set the weights
W = [npr.randn(2, 10), npr.randn(10, 1)]
lmb = 0.001

# Train a neural network
for i in range(50):
    loss_grad = grad(loss_function)(W, x_space, y_space)
    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]

print "Error: " + str(loss_function(W, x_space, y_space))

# Results
surface2 = np.zeros((ny, nx))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        net_outt = neural_network(W, [x, y])[0]
        surface2[i][j] = psy_trial([x, y], net_outt)
print surface[2]
print surface2[2]

# Draw results
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 3)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()
