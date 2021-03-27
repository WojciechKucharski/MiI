import matplotlib.pyplot as plt
import numpy as np
import math
import random
from genClass import *

samples = 10**5

r = generator(type = 1, zk = 2.1)
t = r.rand(samples)

c = autocorr(t, range(0, samples-5))
print(chisquare(t))
plt.plot(c)
plt.show()