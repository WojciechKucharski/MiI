from orto import *
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from functools import cache

lab7 = StaticSystem(sigma=0.01, distr="N") # do wyboru "N" i "C"

def lab7_2(N: int, size: float = 0.2):
    Xn, _, Yn = lab7.simulate(N)
    x = np.linspace(-math.pi, math.pi, 1000)
    plt.plot(x, [m_function(xn) for xn in x])
    plt.scatter(Xn, Yn, s=size)
    plt.legend(["m(x)", "TN"])
    plt.xlabel("Xn")
    plt.ylabel("Yn")
    plt.show()


def lab7_4(L: int, X: List[float], N: int = 500):
    mN = lab7.mN(N=N, L=L, X=X)
    x = np.linspace(-math.pi, math.pi, 1000)
    plt.plot(x, [m_function(xn) for xn in x])
    plt.plot(X, mN)
    plt.legend(["m(x)", "TN"])
    plt.xlabel("Xn")
    plt.ylabel("Yn")
    plt.show()


def lab7_5(L: List[int], Q: int = 100, N: int = 500):
    valid = lab7.valid(N=N, L=L, Q=Q)
    plt.plot(L, valid)
    plt.xlabel("L")
    plt.ylabel("valid(L)")
    plt.show()

D = 12

a = list(np.random.uniform(0, 5, D))

lab8 = MISO(D=D, b=0, sigma_Z=0.2, sigma_X=0.1, a=a)

lab9 = MISO(D=D, b=0.5, sigma_Z=0.2, sigma_X=0.1, a=a)

def lab8_4(N: int):
    lab8.cov1(N)

def lab8_5(N: int, L: int, sigma_Z: List[float]):
    Err = lab8.ErrInSigma(sigma=sigma_Z, N=N, L=L)
    plt.plot(sigma_Z, Err)
    plt.xlabel("sigma")
    plt.ylabel("Err")
    plt.show()

def lab9_4(N: int):
    lab8.cov2(N)

def lab9_5(N: int, L: int, sigma_Z: List[float]):
    Err = lab9.ErrInSigma(sigma=sigma_Z, N=N, L=L)
    plt.plot(sigma_Z, Err)
    plt.xlabel("sigma")
    plt.ylabel("Err")
    plt.show()

#lab7_2(N=10000, size=0.1)
#lab7_4(L=25, X=list(np.linspace(-3, 3, 200)), N=500)
#lab7_5(L=list(range(1,10)), Q=100, N=500)

#lab7_6 to samo co 4 ale wstawić optymalne L
#lab7_7 to samo ale na gorze zmienić "N" na "C"

#lab8_5(N=1000)
#lab8_5(N=500, L=20, sigma_Z=list(np.linspace(0.01,0.2,10)))

