import math
import numpy as np
import matplotlib.pyplot as plt
from generator import *

def histArr(X, nbins = 10, var = False):
    Xmin, Xmax = np.min(X), np.max(X)
    xaxis = np.linspace(Xmin, Xmax, nbins + 1)
    a, b = np.histogram(X, xaxis)
    if var:
        return np.var(nbins * a / len(X))
    return nbins * a / len(X)

def varPlot(samples = 10**5, lin = [1.1, 5.1, 125]):
    t = np.linspace(lin[0], lin[1], lin[2])
    y = []
    for x in t:
        y.append(histArr(r1(samples, z=x), var=True))
    plt.plot(t, y)
    plt.ylabel("Wariancja")
    plt.xlabel("Z")
    plt.show()

def loop(type = 1, maxI = 10**15, lin = [1.5, 5.1, 15]):
    t = np.linspace(lin[0], lin[1], lin[2])
    y = []
    for x in t:
        print(x)
        if type == 1:
            y.append(r1(maxI, x0 = 0.5, z = x, loopBreak=True))
        else:
            y.append(r2(maxI, k = int(x), loopBreak=True))
        print(y[-1])
    plt.plot(t, y)
    plt.ylabel("Okres")
    plt.xlabel("Z")
    plt.show()