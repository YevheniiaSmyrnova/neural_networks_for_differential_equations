import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr

from matplotlib import pyplot as plt


nx = 20
dx = 1. / nx


def f(a, b, c, t):
    """
        dx/dt = at^2+bt+c
        This is f() function on the right
    """
    return a * t ** 2 + b * t + c


def F(a, b, c, t):
    """

    :return:
    """
    return (a * t ** 3) / 3 + (b * t ** 2) / 2 + c * t


def initial_condition(a, b, c, t0=0, x0=0):
    """
    Initial condition for dx/dt = at^2+bt+c
    """
    return x0 - F(a, b, c, t0)


def analytic_solution(a, b, c, t):
    """
        Analytical solution of current problem
    """

    return (a*t**3)/3 + (b*t**2)/2 + c*t + initial_condition(a, b, c)


def nonlin(x, deriv=False):
    if (deriv == True):
        return sigmoid_grad(x)
    return sigmoid(x)


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def sigmoid_grad(t):
    return sigmoid(t) * (1 - sigmoid(t))

# input data
n = 20
T = []
for i in range(0, n+1, 1):
    T.append([1, 1, 1, i*(1./n)])
T = np.array(T)
# print T

# output data
X = []
for i in range(0, n+1, 1):
    X.append(analytic_solution(*T[i]))
X = np.array(X)
# print X

np.random.seed(1)

# 4 input and 1 output
syn0 = 2 * np.random.random(4) - 1

iter_count = 100000
while True:

    l0 = T

    l1 = nonlin(np.dot(l0, syn0))

    l1_error = (X - l1)**2

    l1_delta = l1_error * nonlin(l1, True)  # !!!

    syn0 += 0.0002*np.dot(l0.T, l1_delta)  # !!!

    iter_count -= 1
    print iter_count, l1_error.sum(), l1
    if iter_count == 0:
        break
    elif l1_error.sum() <= 0.00000000002:
        break


print "Output:"
print l1
print X