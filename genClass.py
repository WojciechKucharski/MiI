import math
import matplotlib.pyplot as plt
import numpy as np
import random as randLib
from typing import List

class generator:
    def __init__(self, type: int = 1,
                 x0: float = math.pi,
                 z: float = math.pi ** 3,
                 k: int = 10,
                 distr: int = 0):
        self.distr = distr
        self.x0 = x0 % 1
        if type in [0, 1, 2]:
            self.type = type
        else:
            self.type = 1
        self.z, self.k = z, k

    def rand(self, size: int = 1):
        size = max(size, 1)
        x = eval(f"self.r{self.type + 1}({size})")
        for i in range(len(x)):
            x[i] = eval(f"self.dis{self.distr}(x[i])")
        if size == 1:
            return x[0]
        return x

    def r1(self, size: int):
        tab = [self.x0 % 1]
        for x in range(size):
            new = (tab[-1] * self.z) % 1
            tab.append(new)
        self.x0 = tab[-1]
        return tab[-size:]

    def r2(self, size: int):
        a = []
        for x in range(int(self.k)):
            a.append(math.pi ** (x + 1) % 2)
        tab = [self.x0 % 1]
        for x in range(size + self.k):
            new = 0.0
            for i in range(min(x + 1, int(self.k))):
                new += tab[-i - 1] * a[i]
            tab.append((new + self.x0) % 1)
        self.x0 = tab[-1]
        return tab[-size:]

    def r3(self, size: int):
        tab = []
        for _ in range(size):
            tab.append(randLib.random())
        return tab

    def dis0(self, x: float) -> float:
        return x

    def dis1(self, x: float) -> float:
        return x ** 0.5

    def dis2(self, x: float) -> float:
        return float(x > 0.5) * ((2 * x) ** 0.5 - 1) + float(x <= 0.5) * (1 - (2 - 2 * x) ** 0.5)

    def dis3(self, x: float) -> float:
        return -math.log(1 - x, math.e)

    def dis4(self, x: float, b: float = 1, u: float = 0) -> float:
        if x < 0.5:
            return u + b * (math.log(2 * x, math.e))
        else:
            return u - b * (math.log(2 - 2 * x, math.e))


class reject1:
    def __init__(self, a: float = -1, b: float = 1, d: float = 1, func: str = "1-abs(x)"):
        self.abd = [min(a, b), max(a, b), abs(d)]
        self.f = func

    def evaluate(self, fun: str, x: float) -> float:
        return eval(fun)

    def rand(self, size: int = 1):
        z = []
        while len(z) < size:
            y = randLib.random() * self.abd[2]
            x = self.abd[0] + (self.abd[1] - self.abd[0]) * randLib.random()
            if y < self.evaluate(self.f, x):
                z.append(x)
        if size == 1:
            return z[0]
        return z


class reject2:
    def __init__(self, c: float = (2 * math.e / math.pi) ** 0.5,
                 g: str = "0.5*math.exp(-abs(x))",
                 G: str = "(x<0.5)*(math.log(2*x))+(x>=0.5)*(-math.log(2-2*x))",
                 f: str = "math.exp(-0.5*x**2)*((2*math.pi)**(-0.5))"):
        self.c, self.g, self.G, self.f = c, g, G, f

    def evaluate(self, fun: str, x: float) -> float:
        return eval(fun)

    def rand(self, size: int = 1):
        z = []
        while len(z) < size:
            x = self.evaluate(self.G, randLib.random())
            y = self.c * self.evaluate(self.g, x) * randLib.random()
            if y < self.evaluate(self.f, x):
                z.append(x)
        if size == 1:
            return z[0]
        else:
            return z


class numeric:
    def __init__(self, a: float, b: float, N: int, fun: str):
        self.N = N
        self.calculate(fun, a, b)

    def rand(self, size: int):
        output = []
        for i in range(size):
            output.append(
                self.reversed[math.floor(self.N * randLib.random())]
            )
        return output

    def calculate(self, fun: str, a: float, b: float):
        t = np.linspace(a, b, self.N)  # OX dystrybuanty
        distr = [0]  # Oy dystrybuanty
        for i in range(1, self.N):
            x = a + i * (b - a) / self.N
            distr.append(distr[-1] + eval(fun))
        for i in range(len(distr)):
            distr[i] /= max(distr)  # normalizacja

        maxj = 0
        g = np.linspace(0, 1, self.N)
        for i in range(self.N):
            for j in range(max(0, maxj - 2), self.N - 1):
                if g[i] > distr[j] and g[i] <= distr[j + 1]:
                    maxj = j
                    g[i] = (t[j] + t[j + 1]) / 2
                    break
        g[0] = g[1]
        self.distr = distr
        self.reversed = g


def unit(t: float) -> float:
    return float(t >= 0)


def discrete(x: float, p=[1, 2, 3, 4, 5, 6],
             v=[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]):
    o = 0
    for i in range(min(len(p), len(v))):
        o += unit(x - sum(p[:i])) * (v[i] - v[i - 1] * (i != 0))
    return o


def analyse(data, nbins: int = 100):
    output = np.zeros(nbins)
    n = len(data)
    for x in data:
        output[int(x * nbins)] += 1
    return output


def hist(data, nbins: int = 100):
    data = analyse(data, nbins)
    t = np.linspace(0, 1, nbins)
    plt.stem(t, data)
    plt.show()


def chisquare(data, nbins: int = 100):
    n = len(data)
    data = analyse(data, nbins)
    output = 0.0
    for x in data:
        output += (x - n / nbins) ** 2
    return output * nbins / n


def autocorr(data, dt, m = None, v=None, printprogress=False):
    n = len(data)
    if m is None:
        m = np.mean(data)
    if v is None:
        v = np.var(data)
    if type(dt) is int:
        dt = [dt]
    output = []
    for dx in dt:
        output.append(0)
        for i in range(n):
            output[-1] += 1 / (v * n) * (data[i] - m) * (data[(i + dx) % n] - m)
        if printprogress:
            print(f"{len(output)}/{len(dt)}")
    return output

def histArr(X, nbins=10, var=False):
    Xmin, Xmax = np.min(X), np.max(X)
    xaxis = np.linspace(Xmin, Xmax, nbins + 1)
    a, b = np.histogram(X, xaxis)
    if var:
        return np.var(nbins * a / len(X))
    return nbins * a / len(X)

