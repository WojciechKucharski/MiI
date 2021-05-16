from orto import *
import numpy as np
import math
import matplotlib.pyplot as plt



def Lab7_zad2(N: int, sigma: float = 0.1, type: str = "N"):
    s = StaticSystem(sigma=sigma, distr=type)
    Xn, Zn, Yn = s.simulate(N)
    plt.scatter(Xn, Yn)
    plt.title("Chmura pomiarów")
    plt.ylabel("Yn")
    plt.xlabel("Xn")
    plt.show()

def Lab7_zad4(N: int, M: int, L: int, sigma: float = 0.1, type: str = "N"):
    s = StaticSystem(sigma=sigma, distr=type)
    x = list(np.linspace(-math.pi, math.pi, M))
    plt.plot(x, [s.m(xn) for xn in x])
    plt.plot(x, s.mN(N, L, x))
    plt.xlabel("x")
    plt.legend(["Oryginał", "mN"])
    plt.show()

def Lab7_zad5(N: int, L: List[int], Q: int = 100, sigma: float = 0.1, type: str = "N"):
    s = StaticSystem(sigma=sigma, distr=type)
    plt.stem(L, s.valid(N=N, L=L, Q=Q))
    plt.xlabel("L")
    plt.ylabel("valid(L)")
    plt.show()


Lab7_zad2(N=1000)
Lab7_zad4(N=250, M=100, L=15)
Lab7_zad5(N=250, L=list(range(10)))

Lab7_zad2(N=250, type="C")
Lab7_zad4(N=250, M=100, L=15, type="C")