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

def lab8_5(N: List[int], L: int):
    Err = [lab8.Err(N_, L) for N_ in N]
    plt.plot(N, Err)
    plt.xlabel("N")
    plt.ylabel("Err")
    plt.show()

def lab9_6(N: int):
    lab9.cov2(N)

def lab9_7(N: List[int], L: int):
    Err = [lab9.Err(N_, L) for N_ in N]
    plt.plot(N, Err)
    plt.xlabel("N")
    plt.ylabel("Err")
    plt.show()

def lab8_5_lab9_7(N: List[int], L: int, lab = 8):
    if lab==8:
        Err = [lab8.Err(N_, L) for N_ in N]
    else:
        Err = [lab9.Err(N_, L) for N_ in N]
    plt.plot(N, Err)
    plt.xlabel("N")
    plt.ylabel("Err")
    plt.show()

#lab7_2(N=10000, size=0.1)
#lab7_4(L=25, X=list(np.linspace(-3, 3, 200)), N=500)
#lab7_5(L=list(range(1,10)), Q=100, N=500)

#lab7_6 to samo co 4 ale wstawić optymalne L
#lab7_7 to samo ale na gorze zmienić "N" na "C"

#lab8_4(N=1000)
#lab8_5(N=range(5,25), L=10)

#lab9_6(N=1000)
#lab9_7(N=range(5,25), L=10)