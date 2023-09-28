import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange

train = np.random.uniform(low=2, high=7, size=120)
test = train[101:]

def lagrangeTest(n, s):
    epsilon = np.random.normal(loc=0, scale=s, size=n)
    poly = lagrange(train[:n] + epsilon, np.sin(train[:n]))
    trainE = sum((Polynomial(poly.coef[::-1])(train[:n]) - np.sin(train[:n])) ** 2) / n
    testE = sum((Polynomial(poly.coef[::-1])(test) - np.sin(test)) ** 2) / n
    print("n: ",n, "trainError: ", trainE, "testError: ", testE)

lagrangeTest(15,0)
lagrangeTest(50, 0)
lagrangeTest(100, 0)
print("_________________")
lagrangeTest(30, .2)
lagrangeTest(30, 1)
lagrangeTest(30, 3)
lagrangeTest(30, 7) 