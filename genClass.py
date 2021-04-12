import math
import matplotlib.pyplot as plt
import numpy as np
import random as randLib

class generator:
    def __init__(self, type = 1, x0 = math.pi, zk = math.pi ** 3, distr = 0):
        self.distr = int(distr)
        self.x0 = float(x0) % 1
        if self.x0 in [0.0, 1.0]:
            self.x0 = 1 / 3
        if type in [0, 1, 2]:
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
            new = (tab[-1] * self.zk)%1
            tab.append(new)
        self.x0 = tab[-1]
        return tab[-size:]

    def r2(self, size):
        a = []
        for x in range(int(self.zk)):
            a.append(math.pi**(x+1) % 2)
        tab = [self.x0 % 1]
        for x in range(size + self.zk):
            new = 0.0
            for i in range(min(x + 1, int(self.zk))):
                new += tab[-i - 1] * a[i]
            tab.append((new + self.x0) % 1)
        self.x0 = tab[-1]
        return tab[-size:]

    def r3(self, size):
        tab = []
        for _ in range(size):
            tab.append(randLib.random())
        return tab


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



class reject1:
    def __init__(self, a=-1, b=1, d=1, func="1-abs(x)"):
        self.abd = [min(a, b), max(a, b), abs(d)]
        self.f = func

    def evaluate(self, fun, x):
        return eval(fun)

    def rand(self, size = 1):
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
    def __init__(self, c = (2*math.e/math.pi)**0.5,
                 g = "0.5*math.exp(-abs(x))",
                 G = "(x<0.5)*(math.log(2*x))+(x>=0.5)*(-math.log(2-2*x))",
                 f = "math.exp(-0.5*x**2)*((2*math.pi)**(-0.5))"):
        self.c, self.g, self.G, self.f  = c, g, G, f

    def evaluate(self, fun, x):
        return eval(fun)

    def rand(self, size = 1):
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
    def __init__(self, a, b, N, fun):
        self.N = N
        self.calculate(fun, a, b)

    def rand(self, size):
        output = []
        for i in range(size):
            output.append(
                self.reversed[math.floor(self.N * randLib.random())]
            )
        return output

    def calculate(self, fun, a, b):
        t = np.linspace(a, b, self.N) #OX dystrybuanty
        distr = [0] #Oy dystrybuanty
        for i in range(1, self.N):
            x = a + i * (b - a) / self.N
            distr.append(distr[-1] + eval(fun))
        for i in range(len(distr)):
            distr[i] /= max(distr) #normalizacja

        maxj = 0
        g = np.linspace(0, 1, self.N)
        for i in range(self.N):
            for j in range(max(0, maxj - 2), self.N - 1):
                if g[i] > distr[j] and g[i] <= distr[j + 1]:
                    maxj = j
                    g[i] = (t[j] + t[j + 1])/2
                    break
        g[0] = g[1]
        self.distr = distr
        self.reversed = g


def unit(t):
    return float(t>=0)


def discrete(x, p = [1, 2, 3, 4, 5, 6],
             v = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]):
    o = 0
    for i in range(min(len(p), len(v))):
        o += unit(x - sum(p[:i])) * (v[i] - v[i-1] * (i!=0))
    return o

def analyse(data, nbins = 100):
    output = np.zeros(nbins)
    n = len(data)
    for x in data:
        output[int(x * nbins)] += 1 #nbins / n
    return output

def hist(data, nbins = 100):
    data = analyse(data, nbins)
    t = np.linspace(0, 1, nbins)
    plt.stem(t, data)
    plt.show()

def chisquare(data, nbins = 100):
    n = len(data)
    data = analyse(data, nbins)
    output = 0.0
    for x in data:
        output += (x - n / nbins) ** 2
    return output * nbins / n

def mean(data):
    output = 0.0
    n = len(data)
    for x in data:
        output += x / n
    return output

def var(data, m = None):
    n = len(data)
    if m is None:
        m = mean(data)
    output = 0.0
    for x in data:
        output += (m - x)**2 / n
    return output

def autocorr(data, dt, m = None, v = None, printprogress = False):
    n = len(data)
    if m is None:
        m = mean(data)
    if v is None:
        v = var(data, m)
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

def multi_hist(z=[1.1, 2.2, 3.3]):
    r = generator(type=0)
    fig, axs = plt.subplots(len(z))
    for i in range(len(z)):
        r.zk = z[i]
        axs[i].hist(r.rand(10 ** 5), 25)
        axs[i].set_title(f"z = {z[i]}")
    plt.show()

def histArr(X, nbins = 10, var = False):
    Xmin, Xmax = np.min(X), np.max(X)
    xaxis = np.linspace(Xmin, Xmax, nbins + 1)
    a, b = np.histogram(X, xaxis)
    if var:
        return np.var(nbins * a / len(X))
    return nbins * a / len(X)

#########################################################################################################################################################
def varEstimator(data, biased = False):
    m = mean(data)
    n = len(data)
    output = 0.0
    for x in data:
        output += ((x - m)**2) / (n+1-biased)
    return output