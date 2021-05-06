import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from estimators import *

gen = normal_generator(1, 1)
f = Tester(gen)
x = np.linspace(-1, 3, 100)
plt.plot(f.fN(N=100, X=x, kernel="float(3/4*(1-x**2)) * float(x<=1 and x>=-1)", hN=1))
plt.show()