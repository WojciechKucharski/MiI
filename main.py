from orto import *
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from functools import cache


def m_function_2(x):
    return -x * math.exp(-(x**2))

def m_function(x: float, a: float = 1) -> float:
    x+=1
    if abs(x) > 2:
        return 0
    elif abs(x) < 1:
        return a * x ** 2
    else:
        return 1


def phi_function(k: int, x: float) -> float:
    if k == 0:
        return (2 * math.pi) ** (-0.5)
    else:
        return math.pi ** (-0.5) * math.cos(x * k)

def phi_function_sin(k: int, x: float) -> float:
    if k == 0:
        return (2 * math.pi) ** (-0.5)
    else:
        return math.pi ** (-0.5) * math.sin(x * k)
@cache
def haver(k,x):
    if k == 0:
        return 1
    elif k==1:
        return 2*x
    elif k==2:
        return 4*x**2-2
    else:
        return 2*x*haver(k-1,x)-2*(k-1)*haver(k-2,x)

def mNhaver(X, L, N):
    mN = []
    Xn = list(np.random.uniform(-math.pi, math.pi, N))
    Zn = list(np.random.normal(0, 0.1, N))
    Yn = [Zn[i] + m_function(Xn[i]) for i in range(N)]
    for x in X:
        f,g=0,0
        for k in range(L):
            alfa = float(np.mean([
                       Yn[i] * haver(k, Xn[i]) for i in range(N)]))
            beta = float(np.mean([
                       haver(k, Xn[i]) for i in range(N)]))
            g += alfa * haver(k, x)
            f += beta * haver(k, x)
        if f==0:
            mN.append(0)
        else:
            mN.append(g/f)
    return mN


def mN(X, L, N):
    mN = []
    Xn = list(np.random.uniform(-math.pi, math.pi, N))
    Zn = list(np.random.normal(0, 0.1, N))
    Yn = [Zn[i] + m_function(Xn[i]) for i in range(N)]
    for x in X:
        f,g=0,0
        for k in range(L):
            alfa = float(np.mean([
                       Yn[i] * phi_function(k, Xn[i]) for i in range(N)]))
            beta = float(np.mean([
                       phi_function(k, Xn[i]) for i in range(N)]))
            g += alfa * phi_function(k, x)
            f += beta * phi_function(k, x)
        if f==0:
            mN.append(0)
        else:
            mN.append(g/f)
    return mN

def mNsin(X, L, N):
    mN = []
    Xn = list(np.random.uniform(-math.pi, math.pi, N))
    Zn = list(np.random.normal(0, 0.1, N))
    Yn = [Zn[i] + m_function(Xn[i]) for i in range(N)]
    for x in X:
        f,g=0,0
        for k in range(L):
            alfa = float(np.mean([
                       Yn[i] * phi_function_sin(k, Xn[i]) for i in range(N)]))
            beta = float(np.mean([
                       phi_function_sin(k, Xn[i]) for i in range(N)]))
            g += alfa * phi_function_sin(k, x)
            f += beta * phi_function_sin(k, x)
        if f==0:
            mN.append(0)
        else:
            mN.append(g/f)
    return mN

def mNboth(X, L, N):
    mN = []
    Xn = list(np.random.uniform(-math.pi, math.pi, N))
    Zn = list(np.random.normal(0, 0.01, N))
    Yn = [Zn[i] + m_function(Xn[i]) for i in range(N)]
    for x in X:
        f,g=0,0
        g2,f2=0,0
        for k in range(L):
            alfa = float(np.mean([
                       Yn[i] * phi_function_sin(k, Xn[i]) for i in range(N)]))
            beta = float(np.mean([
                       phi_function_sin(k, Xn[i]) for i in range(N)]))
            g += alfa * phi_function_sin(k, x)
            f += beta * phi_function_sin(k, x)

            alfa2 = float(np.mean([
                Yn[i] * phi_function(k, Xn[i]) for i in range(N)]))
            beta2 = float(np.mean([
                phi_function(k, Xn[i]) for i in range(N)]))
            g2 += alfa2 * phi_function(k, x)
            f2 += beta2 * phi_function(k, x)
        if f==0 or f2 == 0:
            mN.append(0)
        else:
            mN.append(g/f+g2/f2)
    return mN



x = np.linspace(-math.pi, 1, 100)
plt.plot(x, [m_function(X) for X in x])

plt.plot(x, mNboth(x, 12, 100))
plt.xlabel("Xn")
plt.ylabel("Yn")
plt.legend(["m(x)", "cos&sin"])
plt.show()