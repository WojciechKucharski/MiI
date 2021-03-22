import matplotlib.pyplot as plt
import numpy as np
import math
import random as r
from genClass import *

samples = 10**5

R = generator()
plt.hist(R.rand(samples), 100)
plt.title("Generator U[0,1]")
plt.show()

E = reject(a = -1, b = 1, d = 1, fun = "1-abs(x)")
plt.hist(E.rand(samples), 100)
plt.title("Zadanie 1")
plt.show()

U = reject(a = 0, b = 1, d = 50, fun = "Funkcja_zadanie_2(x)")
plt.hist(U.rand(samples/100), 100)
plt.title("Zadanie 2")
plt.show()

G = reject(a = -1, b = 1, d = 1, fun = "(1-x*x)**0.5")
plt.hist(G.rand(samples), 100)
plt.title("Zadanie 3")
plt.show()

D = reject2(c = (2*math.e/math.pi)**0.5, dis = 4,
                 fun = "math.exp(-0.5*x**2)*((2*math.pi)**(-0.5))")
plt.hist(D.rand(samples), 100)
plt.title("Zadanie 4")
plt.show()


