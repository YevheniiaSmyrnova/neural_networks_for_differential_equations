"""
Solution of ODE second order for boundary value
Error: 0.003685685168718363
"""
import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr
from matplotlib import pyplot as plt

a = 0
b = 1
A = 0.
B = np.exp(-1. / 5.) * np.cos(1.)
nx = 20
dx = 1. / nx


def f(x, psi, dpsi):
    """
    d2(psi)/dx^2 = f(x, dpsi/dx, psi)
    This is f() function on the right
    :param x:
    :param psi:
    :param dpsi:
    :return:
    """
    return -1. / 5. * np.exp(-x / 5.) * np.cos(x) - 1. / 5. * dpsi - psi


def psi_analytic(x):
    """
    Analytical solution of current problem
    :param x:
    :return:
    """
    return np.exp(-x / 5.) * np.sin(x)


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


def psi_trial(xi, net_out):
    """
    Trial solution
    :param xi:
    :param net_out:
    :return:
    """
    return A * (1 - xi) + B * xi + xi * (1 - xi) * net_out


psi_grad = grad(psi_trial)
psi_grad2 = grad(psi_grad)


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
        psi_t = psi_trial(xi, net_out)
        gradient_of_trial = psi_grad(xi, net_out)
        second_gradient_of_trial = psi_grad2(xi, net_out)
        func = f(xi, psi_t, gradient_of_trial)
        err_sqr = (second_gradient_of_trial - func) ** 2
        loss_sum += err_sqr
    return loss_sum


# Set the weights
W = [npr.randn(1, 10), npr.randn(10, 1)]
lmb = 0.001

x_space = np.linspace(0, 2, nx)
y_space = psi_analytic(x_space)

# Train a neural network
for i in range(1000):
    loss_grad = grad(loss_function)(W, x_space)
    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]

# Results
res = [psi_trial(xi, neural_network(W, xi)[0][0]) for xi in x_space]

print "Error: " + str(loss_function(W, x_space))

# Draw results
plt.figure()
plt.plot(x_space, y_space)
plt.plot(x_space, res)
plt.show()
