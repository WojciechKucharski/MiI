import math
import matplotlib.pyplot as plt
import numpy as np

def histArr(X, nbins = 10, var = False):
    Xmin, Xmax = np.min(X), np.max(X)
    xaxis = np.linspace(Xmin, Xmax, nbins + 1)
    a, b = np.histogram(X, xaxis)
    if var:
        return np.var(nbins * a / len(X))
    return nbins * a / len(X)

def Funkcja_zadanie_2(x):
    if x < 1/100:
        return 50
    else:
        return 50/99

class generator:
    def __init__(self, type = 1, x0 = math.pi, zk = math.pi ** 3, distr = 0):
        self.distr = int(distr)
        self.x0 = float(x0) % 1
        if self.x0 in [0.0, 1.0]:
            self.x0 = 1 / 3
        if type in [0, 1]:
            self.type = type
        else:
            self.type = 1
        if self.type == 1:
            self.zk = int(zk)
        else:
            self.zk = float(zk)

    def rand(self, size = 1):
        size = int(size)
        if size <= 0:
            size = 1
        x = eval(f"self.r{self.type + 1}({size})")
        for i in range(len(x)):
            x[i] = eval(f"self.dis{self.distr}(x[i])")
        if size == 1:
            return x[0]
        return x

    def r1(self, size):
        tab = [self.x0 % 1]
        for x in range(size):
            tab.append(tab[-1] * self.zk - math.floor(tab[-1] * self.zk))
        self.x0 = tab[-1]
        return tab[-size:]

    def r2(self, size):
        a = []
        for x in range(int(self.zk)):
            a.append(math.pi**(x+1) % 2)
        tab = [self.x0 % 1]
        for x in range(size):
            new = 0.0
            for i in range(min(x + 1, int(self.zk))):
                new += tab[-i - 1] * a[i]
            tab.append((new + self.x0) % 1)
        self.x0 = tab[-1]
        return tab[-size:]


    def dis0(self, x):
        return x

    def dis1(self, x):
        return x**0.5

    def dis2(self, x):
        if x < 0.5:
            return (2*x)**0.5 -1
        else:
            return 1 - (2-2*x)**0.5

    def dis3(self, x):
        return -math.log(1-x, math.e)

    def dis4(self, x, b = 1, u = 0):
        if x < 0.5:
            return u+b*(math.log(2*x, math.e))
        else:
            return u - b*(math.log(2-2*x, math.e))



class reject: #TODO
    def __init__(self, a = -1, b = 1, d = 1, fun = "1-abs(x)"):
        self.a = min(a, b)
        self.b = max(a, b)
        self.d = d
        self.fun = fun
        self.g = generator(type = 0, zk = 123.45, x0 = 0.345)

    @property
    def dx(self):
        return self.a + (self.b - self.a) * self.g.rand()

    def rand(self, size = 1):
        z = []
        while len(z) < size:
            y = self.g.rand() * self.d
            x = self.dx
            if y < eval(self.fun):
                z.append(x)
        if size == 1:
            return z[0]
        return z


class reject2: #TODO
    def __init__(self,
                 c = (2*math.e/math.pi)**0.5, dis = 4,
                 fun = "math.exp(-0.5*x**2)*((2*math.pi)**(-0.5))"):
        self.c = c
        self.fun = fun
        self.dis = dis
        self.n = generator(type = 0, zk = 73.45, x0 = 0.355, distr = dis)
        self.g = generator(type = 1, zk = 93.45, x0 = 0.769)

    def rand(self, size = 1):
        z = []

        while len(z) < size:

            x = self.n.rand()
            y = self.c * 0.5 * math.exp(-abs(x)) * self.g.rand()
            if y < eval(self.fun):
                z.append(x)

        if size == 1:
            return z[0]
        else:
            return z




















