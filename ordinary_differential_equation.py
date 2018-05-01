"""
Solution of ODE
Error: 0.18696873538824116
"""
import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr
from matplotlib import pyplot as plt

a = 0
b = 1
A = 1.
nx = 20
dx = 1. / nx


def f(x, psi):
    """
    d(psi)/dx = f(x, psi)
    This is f() function on the right
    :param x:
    :param psi:
    :return:
    """
    return x ** 3 + 2. * x + x ** 2 * (
    (1. + 3. * x ** 2) / (1. + x + x ** 3)) - psi * (
    x + (1. + 3. * x ** 2) / (1. + x + x ** 3))


def psi_analytic(x):
    """
    Analytical solution of current problem
    :param x:
    :return:
    """
    return (np.exp((-x ** 2) / 2.)) / (1. + x + x ** 3) + x ** 2


def sigmoid(x):
    """
    Sigmoid function
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    """
    Derivative of sigmoid function
    :param x:
    :return:
    """
    return sigmoid(x) * (1 - sigmoid(x))


def neural_network(W, x):
    """
    Neural network
    :param W:
    :param x:
    :return:
    """
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])


def d_neural_network_dx(W, x, k=1):
    """
    Derivative of neural network
    :param W:
    :param x:
    :param k:
    :return:
    """
    return np.dot(np.dot(W[1].T, W[0].T ** k), sigmoid_grad(x))


def loss_function(W, x):
    """
    Loss function
    :param W:
    :param x:
    :return:
    """
    loss_sum = 0.
    for xi in x:
        net_out = neural_network(W, xi)[0][0]
        psi_t = A + xi * net_out
        d_net_out = d_neural_network_dx(W, xi)[0][0]
        d_psi_t = net_out + xi * d_net_out
        func = f(xi, psi_t)
        err_sqr = (d_psi_t - func) ** 2
        loss_sum += err_sqr
    return loss_sum


# Set the weights
W = [npr.randn(1, 10), npr.randn(10, 1)]
lmb = 0.001

x_space = np.linspace(a, b, nx)
y_space = psi_analytic(x_space)

# Train a neural network
for i in range(1000):
    loss_grad = grad(loss_function)(W, x_space)
    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]

# Results
res = [1 + xi * neural_network(W, xi)[0][0] for xi in x_space]

print "Error: " + str(loss_function(W, x_space))

# Draw results
plt.figure()
plt.plot(x_space, y_space)
plt.plot(x_space, res)
plt.show()
