import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from estimators import *

gen = normal_generator(1, 1)
f = Tester(gen)

x = np.linspace(-1, 3, 250)
plt.plot(
    x, [f.CDF(x_n) for x_n in x],
    x, f.FN(N=15, X=x)
)
plt.show()

plt.plot(
    x, [f.PDF(x_n) for x_n in x],
    x, f.fN(N=15, X=x, hN=2)
)
plt.show()

