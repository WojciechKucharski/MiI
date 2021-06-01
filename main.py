from orto import *
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from functools import cache

obj = MISO(D=10, a=np.linspace(0,44,10))

print(obj.ErrInSigma(np.linspace(0.1,2,10), 25, 11))