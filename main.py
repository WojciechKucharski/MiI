from generator import *
import matplotlib.pyplot as plt
from functions import *
import numpy as np
import math
import random as r
def double(samples = 10**5, type = 0):
    if type == 0:
        X = np.linspace(0, 1, samples)
    if type == 1:
        X = r1(samples, z = 1.1)
    if type == 2:
        X = r2(samples)
    if type == 3:
        X = np.random.rand(samples)

    W = []
    for i in range(len(X) - 1):
        W.append(X[i] * X[i + 1])

    return X


for i in range(4):
    plt.hist(double(type = i, samples=10**5))
    plt.show()





















"""
Y = []
for x in X:
    if x < 0.5:
        Y.append(math.sqrt(2*x) - 1)
    else:
        Y.append(-math.sqrt(2-2*x)+1)

Z = []

for x in X:
    Z.append(-math.log(math.e, 1 - x))
"""