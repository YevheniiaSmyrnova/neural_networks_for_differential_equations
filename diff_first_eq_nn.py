"""
NN for resolving first order differential equation

psi - function to find
x - variable

ODE - Ordinary differential equation

d(psi(x))/dx + (x + ( 1 +  3 * x**2) / (1 + x + x**3) ) * psi(x) = x**3 + 2 * x + x**2 * ( (1 + 3 * x**2) / ( 1 + x + x**3))

of if we move d(psi(x))/dx to right side and other to left side

d(psi(x))/dx  = x**3 + 2 * x + x**2 * ( (1 + 3 * x**2) / ( 1 + x + x**3)) - (x + ( 1 +  3 * x**2) / (1 + x + x**3) ) * psi(x)

Redeclare next:
x + ( 1 +  3 * x**2) / (1 + x + x**3)  as A(x) part of equation
x**3 + 2 * x + x**2 * ( (1 + 3 * x**2) / ( 1 + x + x**3)) as B(x) part of equation

Will get:  d(psi(x))/dx = B(x) - A(x) * psi(x)

And redeclare B(x) - A(x) * psi(x) as F(x, psi(x))

Will get: d(psi(x))/dx = F(x, psi(x))
So, to solve ODE we must to find derivative of F(x, psi(x)) function.
"""
import numpy as np
from autograd import grad
from matplotlib import pyplot as plt

number_of_points = 100  # number of points in which we want to find a solution
interval = [0.0, 1.0]  # interval in which we must to solve equation
step_size = interval[1] / number_of_points  # dx or difference between current and previous point

psi_0 =

def A(x):
    """
    Right part of initial equation
    x + ( 1 +  3 * x**2) / (1 + x + x**3)
    """
    return x + (1. + 3.*x**2) / (1. + x + x**3)


def B(x):
    """
    Right part of initial equation
    x**3 + 2 * x + x**2 * ( (1 + 3 * x**2) / ( 1 + x + x**3))
    """
    return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))


def F(x, psi):
    """
    d(psi(x))/dx = F(x, psi(x))
    """
    return B(x) - A(x) * psi


def psi_analytic_solution(x):
    """
    Analytical solution of current ODE.
    """
    return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2


x_space = np.linspace(interval[0], interval[1], number_of_points)
y_space = psi_analytic_solution(x_space)
psi_fd = np.zeros_like(y_space)
psi_fd[0] = 1.  # boundary conditions

# finite differences -  psi(x[i]) = psi(x[i-1]) + F(x[i], psi(x[i-1])) * step_size
for i in range(1, len(x_space)):
    psi_fd[i] = psi_fd[i - 1] + F(x_space[i],  psi_fd[i - 1]) * step_size


########################## NN ######################################

def activation_function(x):
    return 1 / (1 + np.exp(-x))


def activation_function_derivative(x):
    return grad(activation_function)(x)


def neural_network(weights, x):
    a1 = activation_function(np.dot(x, weights[0]))
    return np.dot(a1, weights[1])


def d_neural_network_dx(W, x, k=1):
    return np.dot(np.dot(W[1].T, W[0].T**k), sigmoid_grad(x))


def loss_function(W, x):
    loss_sum = 0.
    for xi in x:
        net_out = neural_network(W, xi)[0][0]
        psy_t = 1. + xi * net_out
        d_net_out = d_neural_network_dx(W, xi)[0][0]
        d_psy_t = net_out + xi * d_net_out
        func = f(xi, psy_t)
        err_sqr = (d_psy_t - func)**2

        loss_sum += err_sqr
    return loss_sum


W = [np.random.randn(1, 10), np.random.randn(10, 1)]
lmb = 0.001

# x = np.array(1)
# print neural_network(W, x)
# print d_neural_network_dx(W, x)

for i in range(1000):
    loss_grad = grad(loss_function)(W, x_space)

    #     print loss_grad[0].shape, W[0].shape
    #     print loss_grad[1].shape, W[1].shape

    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]

# print loss_function(W, x_space)


plt.figure()
plt.plot(x_space, y_space)
plt.plot(x_space, psi_fd)
plt.show()
