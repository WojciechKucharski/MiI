from orto import *
import numpy as np
import math
import matplotlib.pyplot as plt

s = StaticSystem(sigma=0.1, a=1)


x = np.linspace(-math.pi, math.pi, 1000)
Xn, Zn, Yn = s.simulate(1000)
plt.plot(x, [s.m(xn) for xn in x])
plt.plot(x, s.mN(100, 10, x))
plt.show()


