import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from estimators import *

g = staticDynamicSystem()
h = np.linspace(-7, 7, 230)
x = g.mN(hN=1, N=9200, X=h)
plt.plot(h, x)
plt.show()
