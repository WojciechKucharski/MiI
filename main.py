import matplotlib.pyplot as plt
import numpy as np
import math
import random
from genClass import *
import time

samples = 10**3

z = 300000


r = generator(type=0, zk=z)

T = np.linspace(10**-5, 1 - 10**-5, 10**5)
t = r.rand(samples)

y = []
for x in T:
    y.append(z * x - math.floor(z * x))

plt.scatter(t[:-1], t[1:])
#plt.plot(T, y)
plt.show()