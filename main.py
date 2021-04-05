import matplotlib.pyplot as plt
import numpy as np
import math
import random
from genClass import *
import time

def unit(t):
    return float(t>=0)




def discrete(x, p, v):
    o = 0
    for i in range(min(len(p), len(v))):
        o += unit(x - sum(p[:i])) * (v[i] - v[i-1] * (i!=0))
    return o

p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
v = [1, 2, 3, 4, 5, 6]
y = []
for x in range(10**5):
    y.append(discrete(random.random(), p, v))
plt.hist(y, 100)
plt.show()