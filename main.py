import math
import numpy as np
import matplotlib.pyplot as plt

def r1(x0 = 0.1, z = 3.14, size = 5):
    tab = [x0]
    for _ in range(size+1):
        tab.append(tab[-1]*z - math.floor(tab[-1]*z))
    return tab[2:]

def hist(x0 = 0.1, z = 3.14, size = 10**5):
    plt.hist(r1(x0, z, size))
    plt.title("x0 = "+str(x0) + ", z = "+str(z))
    plt.show()

def histArr(X, nbins = 10, var = False):
    Xmin, Xmax = np.min(X), np.max(X)
    xaxis = np.linspace(Xmin, Xmax, nbins + 1)
    a, b = np.histogram(X, xaxis)
    if var:
        return np.var(a/len(X))
    return a/len(X)

def varDraw(start = 0, stop = 15, samples = 10**3, subsamples = 10**2, other = 0.5, zMain = True):
    t = np.linspace(start, stop, samples) #create OX
    print(t)
    y = []
    for x in t: #create OY
        if zMain:
            r = r1(other, x, subsamples)
        else:
            r = r1(x, other, subsamples)
        y.append(histArr(r, 25, True))
    plt.plot(t, y)
    plt.ylabel("Wariancja")
    if zMain:
        plt.xlabel("z")
        plt.title("Wariancja rozkładu: x0 = " + str(other) + ", z = OX")
    else:
        plt.xlabel("x0")
        plt.title("Wariancja rozkładu: z = " + str(other) + ", x0 = OX")
    plt.show()

varDraw()