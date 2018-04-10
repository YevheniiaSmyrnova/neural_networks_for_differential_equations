"""
3 layer
"""
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
    X.append([analytic_solution(*T[i])])
X = np.array(X)

np.random.seed(1)

# 4 input and 1 output
syn0 = 2 * np.random.random((4, 5)) - 1
syn1 = 2 * np.random.random((5, 1)) - 1

iter_count = 100000
while True:

    l0 = T

    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l2_error = (X - l2)**2

    if (iter_count % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    l2_delta = l2_error * nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 -= l1.T.dot(l2_delta)
    syn0 -= l0.T.dot(l1_delta)

    iter_count -= 1
    print iter_count, np.mean(np.abs(l2_error)), l2
    if iter_count == 0:
        break
    elif np.mean(np.abs(l2_error)) <= 0.00000000002:
        break


print "Output:"
print l2