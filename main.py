import matplotlib.pyplot as plt
import numpy as np
import math
import random
from genClass import *
import time

def normal(mu, var, N):
    return np.random.normal(mu, math.sqrt(var), N)
def cauchy(mu, var, N):
    o = []
    for x in range(N):
        new = mu + math.sqrt(var) * math.tan(math.pi * (random.random() - 0.5))
        o.append(new)
    return o

def estimate(mu, var, N, type = 0):
    if type in [0, 1, 2]:
        s = normal(mu, var, N)
    else:
        s = cauchy(mu, var, N)
    if type in [0, 3]:
        return mean(s) - mu
    if type in [1, 4]:
        return varEstimator(s, True) - var
    if type in [2, 5]:
        return varEstimator(s, False) - var


def test(N, L, mu, var, type = 0):
    y = []
    for i in N:
        y.append(0)
        for j in range(L):
            y[-1] += (1 / L) * estimate(mu, var, i, type) ** 2
    return y


mu = 0
var = 1
N = 100000
s = normal(mu, var, N)

y = []
t = np.linspace(-5, 5, 100)

for x in t:
    y.append(0)
    for i in range(N):
        y[-1] += float(s[i] <= x) / N

plt.stem(t, y)
plt.show()
