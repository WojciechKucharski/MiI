import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from estimators import *
seed = 12343

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

def Lab4_2(N, X, gen, name="test"):
    plt.plot(X, g[gen].FN(N,X), X, g[gen].CDF(X))
    plt.xlabel("x")
    plt.ylabel("FN")
    plt.title(f"Dystrybuanta empiryczna, {name} ")
    plt.legend(["Estymacja", "Oryginał"])
    plt.show()


"""Lab4_2(100, np.linspace(-0.1,1.1,10000), 1, "Rozkład Normalny")
Lab4_2(100, np.linspace(-0.1,1.1,10000), 4, "Rozkład Jednostajny")
Lab4_2(100, np.linspace(-0.1,1.1,10000), 5, "Rozkład Trójkątny")
Lab4_2(100, np.linspace(-0.1,1.1,10000), 7, "Rozkład Laplace'a")
Lab4_2(100, np.linspace(-0.1,1.1,10000), 8, "Rozkład Exp.")
Lab4_2(100, np.linspace(-0.1,1.1,10000), 9, "Rozkład \"Trudny\"")"""

for i in range(1,11):
    Lab4_2(100, np.linspace(-0.1, 1.1, 10000), i, str(i))