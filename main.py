import matplotlib.pyplot as plt
import numpy as np
import math
import random as r
from genClass import *

c = reject2()

plt.hist(c.rand(10**5), 150)
plt.show()