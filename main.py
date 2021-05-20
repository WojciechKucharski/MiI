from orto import *
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from functools import cache


def Lab7_zad2(s, N: int):
    Xn, Zn, Yn = s.simulate(N)
    plt.scatter(Xn, Yn)
    plt.title("Chmura pomiarów")
    plt.ylabel("Yn")
    plt.xlabel("Xn")
    plt.show()

def Lab7_zad4(s, N: int, M: int, L: int):
    x = list(np.linspace(-math.pi, math.pi, M))
    plt.plot(x, [s.m(xn) for xn in x])
    plt.plot(x, s.mN(N, L, x))
    plt.xlabel("x")
    plt.legend(["Oryginał", "mN"])
    plt.show()

def Lab7_zad5(s, N: int, L: List[int], Q: int = 100):
    plt.stem(L, s.valid(N=N, L=L, Q=Q))
    plt.xlabel("L")
    plt.ylabel("valid(L)")
    plt.show()


def m(x: float) -> float:
    if x > 2 or x < -2:
        return 0
    if 2 >= x > 1 or -2 <= x < -1:
        return 1
    else:
        return x ** 2


def phi(k: int, x: float):
    if k == 0:
        return (2 * math.pi) ** (-1 / 2)
    else:
        return math.cos(k * x) * math.pi ** (-1 / 2)

@cache
def H(k: int, x: float):
    if k == 0:
        return 1
    if k == 1:
        return 2 * x
    if k == 2:
        return 4 * x ** 2 - 2
    else:
        return 2 * x * H(k - 1, x) - 2 * (k - 1) * H(k - 2, x)


s = StaticSystem(m, phi, sigma=0.1, distr="N")

#Lab7_zad2(s, N=1000)
#Lab7_zad4(s, N=250, M=100, L=15)
#Lab7_zad5(s, N=250, L=list(range(1, 10)))

L = range(1, 20)
plt.stem(L, s.plot_alfa_k(N=250, L=L))
plt.xlabel("k")
plt.ylabel("a_k")
plt.show()
