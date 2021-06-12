from orto import *
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from functools import cache

from orto import *
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from functools import cache

lab10 = SISO(b=[1, 2, 3], sigma_U=0.2)

# chmura
Un, Zn, Yn = lab10.simulate(N=1000)
plt.scatter(Un, Yn)


