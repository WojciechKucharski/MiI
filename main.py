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