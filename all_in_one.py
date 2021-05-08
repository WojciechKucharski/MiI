import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from estimators import *
seed = 1234

g = [
    file_generator(), #0
    normal_generator(0,1),#1
    normal_generator(1,1),#2
    normal_generator(0,5),#3
    unit_generator(),#4
    triangle_generator(),#5
    symetric_triangle_generator(),#6
    laplace_generator(),#7
    exp_generator(),#8
    annoying_generator(),#9
    cauchy_generator(0,1)#10
]

def Lab3_1(N, gen):
    plt.plot(N, g[gen].estimator("u", N))
    plt.xlabel("N")
    plt.ylabel("Średnia")
    plt.title("Esytmator średniej")
    plt.show()

    np.random.seed(2323)
    plt.plot(N, g[gen].estimator("s",N))
    np.random.seed(2323)
    plt.plot(N, g[gen].estimator("S", N))
    plt.xlabel("N")
    plt.ylabel("Wariancja")
    plt.title("Estymatory wariancji")
    plt.legend(["Obciążony sN", "Nieobciążony SN"])
    plt.show()

def Lab3_2(N, L, gen):
    plt.plot(N, g[gen].Err("u", 0, N, L))
    plt.xlabel("N")
    plt.ylabel("Err")
    plt.title(f"Błąd empiryczny estymatora średniej, L = {L}")
    plt.show()
    np.random.seed(seed)
    plt.plot(N, g[gen].Err("s", 1, N, L))
    np.random.seed(seed)
    plt.plot(N, g[gen].Err("S", 1, N, L))
    plt.xlabel("N")
    plt.ylabel("Wariancja")
    plt.title(f"Błąd empiryczny estymatora wariancji, L = {L}")
    plt.legend(["Obciążony sN", "Nieobciążony SN"])
    plt.show()

def Lab4_2(N, X, gen):
    plt.plot(X, g[gen].FN(N,X), X, g[gen].CDF(X))
    plt.xlabel("x")
    plt.ylabel("FN")
    plt.title("Dystrybuanta empiryczna")
    plt.legend(["Estymacja", "Oryginał"])
    plt.show()

def Lab4_3(N, X, gen):
    plt.plot(N, g[gen].DN(N,X))
    plt.xlabel("N")
    plt.ylabel("DN")
    plt.title("Supremum błędu dystrybuanty empirycznej")
    plt.show()

def Lab4_4(N, X):
    plt.plot(X, g[0].FN(N,X), X, g[1].CDF(X), X, g[2].CDF(X), X, g[3].CDF(X), X, g[10].CDF(X))
    plt.xlabel("x")
    plt.ylabel("FN")
    plt.title(f"Plik, N = {N}")
    plt.legend(["Plik", "N(0,1)", "N(0,5)", "C(0,1)"])
    plt.show()

def Lab4_5(N,X,L,gen):
    plt.plot(X,g[gen].varFN(N, X, L))
    plt.xlabel("x")
    plt.ylabel("Var{Fn}")
    plt.title("Var{Fn}, L = " + str(L))
    plt.show()

def Lab5_2(N, X, h, gen):
    legend = []
    for hn in h:
        np.random.seed(seed)
        plt.plot(X,g[gen].fN(N,X,kernels["t"],hn))
        legend.append(f"hN = {hn}")
    plt.xlabel("x")
    plt.ylabel("fN")
    plt.title("Jądrowy estymator gęstości prawdopodobieństwa")
    plt.legend(legend)
    plt.show()

def Lab5_3(N, X,hn, kers, gen):
    legend = []
    for x in kers:
        np.random.seed(seed)
        plt.plot(X,g[gen].fN(N,X,kernels[x],hn))
        legend.append(x)
    plt.xlabel("x")
    plt.ylabel("fN")
    plt.title(f"Jądrowy estymator gęstości prawdopodobieństwa, hn = {hn}")
    plt.legend(legend)
    plt.show()

def Lab5_4(N, X, L, h, gen):
    plt.plot(h, [g[gen].ErrfN(N, X, L, hN=hN) for hN in h])
    plt.xlabel("hN")
    plt.ylabel("Err{fN}")
    plt.title("Błąd empiryczny")
    plt.show()

def Lab6_2(N, a, sigma):
    Xn, Zn, Yn = staticSystem(a=a,sigma=sigma).simulate(N)
    plt.scatter(Xn,Yn)
    plt.xlabel("Xn")
    plt.ylabel("Yn")
    plt.title("Chmura pomiarów TN")
    plt.show()

def Lab6_3(N, X, h, a, gen):
    s = staticSystem(g[gen], a=a)
    legend = []
    legend.append("Oryginał")
    plt.plot(X,s.m_fun(X))
    for hn in h:
        np.random.seed(seed)
        plt.plot(X,s.mN(N,X,hn))
        legend.append(f"hN = {hn}")
    plt.xlabel("x")
    plt.ylabel("mN")
    plt.title("Jądrowy estymator funkcji regresji")
    plt.legend(legend)
    plt.show()

def Lab6_4(N, X,hn,  kers, a, gen):
    s = staticSystem(g[gen], a=a)
    legend = []
    legend.append("Oryginał")
    plt.plot(X, s.m_fun(X))
    for x in kers:
        np.random.seed(seed)
        plt.plot(X,s.mN(N,X,hn,kernels[x]))
        legend.append(x)
    plt.xlabel("x")
    plt.ylabel("mN")
    plt.title(f"Jądrowy estymator funkcji regresji, hn = {hn}")
    plt.legend(legend)
    plt.show()

def Lab6_5(N, h, a, gen, Q=100, kernel = kernels["r"]):
    s = staticSystem(g[gen], a=a)
    plt.plot(h, s.valid(h, N, Q, kernel))
    plt.xlabel("h")
    plt.ylabel("valid(h)")
    plt.title(f"Q = {Q}")
    plt.show()

"""
#LAB 3
#zad1
Lab3_1(range(1,15), 1)

#zad2
Lab3_2(range(1,15), 10, 1)

#zad3
Lab3_1(range(1,15), 10)
Lab3_2(range(1,15), 10, 10)

#LAB 4
#zad2
Lab4_2(100, np.linspace(-0.5, 1.5, 100), 5)

#zad3
Lab4_3(range(1,25), np.linspace(0.01,0.99,25), 5)

#zad4
Lab4_4(200, np.linspace(-3,3,200))

#zad5
Lab4_5(100, np.linspace(-3,3,100),15,1)

#LAB 5 DŁUGO SIĘ ROBI
#zad2
Lab5_2(100, np.linspace(-3,3,100), [0.1, 0.2, 0.5, 1.0], 1)

#zad3
Lab5_3(100, np.linspace(-3,3,100), 0.5, ["rectangle", "triangle", "gauss", "epechnikov"], 1)

#zad4
Lab5_4(100, np.linspace(-3,3,100), 5, np.linspace(0.1,1,10), 1)

#LAB 6
#zad2
Lab6_2(1000, 1, 0.01)

#zad3
Lab6_3(100, np.linspace(-2,2,100), [0.1, 0.2, 0.5, 1.0], 1,1)
"""
#zad4
Lab6_4(100, np.linspace(-2,2,100), 0.5, ["rectangle", "triangle", "gauss", "epechnikov"],1, 1)

#zad5
#Lab6_5(10000, np.linspace(0.1,1,10), 1, 1, Q=100)

"""
#zad6 powtórzyć zad4, ale jako hN podać optymalną wartość z zad5
optymalne_hN = 0.33 #PODAĆ
Lab6_4(100, np.linspace(-2,2,100), optymalne_hN, ["rectangle", "triangle", "gauss", "epechnikov"],1, 1)

#zad7
Lab6_3(100, np.linspace(-2,2,100), [0.1, 0.2, 0.5, 1.0], 1,10)
Lab6_4(100, np.linspace(-2,2,100), 0.5, ["rectangle", "triangle", "gauss", "epechnikov"],1, 10)"""