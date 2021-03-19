from generator import *
import matplotlib.pyplot as plt
from functions import *
import numpy as np
import math
import random as r


x = np.linspace(0.00001, 0.9999999, 10**4)

x = rozk4(x, b=9, u =19)

plt.hist(x, 50)
plt.show()












"""
Y = []
for x in X:
    if x < 0.5:
        Y.append(math.sqrt(2*x) - 1)
    else:
        Y.append(-math.sqrt(2-2*x)+1)

Z = []

for x in X:
    Z.append(-math.log(math.e, 1 - x))
"""