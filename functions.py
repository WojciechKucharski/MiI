import math
import numpy as np
import matplotlib.pyplot as plt

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

def spectrum(samples = 10**3, basic = False)

    r2 = r2(0.5, 1.12, samples, 15)
    for i in range(samples):
    r1[i] -= 0.5
    r2[i] -= 0.5

X1 = np.fft.fft(r1)
plt.plot(abs(X1)[3:int(samples/2)])
plt.show()

"""
X2 = np.fft.fft(r2)
plt.plot(abs(X2)[3:int(samples/2)])
plt.show()

def r1(x0 = 0.1, z = 3.14, size = 5):
    tab = [x0]
    for _ in range(size+1):
        tab.append(tab[-1]*z - math.floor(tab[-1]*z))
    return tab[2:]

def r2(x0 = 0.1, z = 3.14, size = 5, k = 3, c = 0.5, m = 1):

    a = r1(x0, z, k)
    tab = [c % m]
    for x in range(size + 1):
        new = 0.0
        for i in range(min(x + 1, k)):
            new += tab[-i-1] * a[i]
        new = (new+c) % m
        tab.append(new)
    return tab[2:]
"""