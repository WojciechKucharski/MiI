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


def impulse(a=0.5, b=0.5, N=100, sigma=0.1):
    lab11 = SISO2(theta=[a, b], sigma_Z=sigma)
    U = [0] * N
    U[0] = 1
    UN, fiN, ZN, YN = lab11.simulate(N=len(U), UN=U)
    plt.plot(range(N), UN, range(N), YN)
    plt.legend(["U", "Y"])
    plt.xlabel("N")
    plt.show()


def err(N, L, a=0.2, b=0.2, sigma=0.1):
    lab11 = SISO2(theta=[a, b], sigma_Z=sigma)
    seed = 1000
    np.random.seed(seed)
    Err = [lab11.Err(N_, L) for N_ in N]
    np.random.seed(seed)
    ErrIV = [lab11.Err(N_, L, True) for N_ in N]
    plt.plot(N, Err, N, ErrIV)
    plt.xlabel("N")
    plt.ylabel("Err")
    plt.legend(["MNK", "MZI"])
    plt.show()


#impulse(a=2, b=0.5, N=1000, sigma=0.01)
err(range(25, 250), L=25, a=0.5, b=0.5, sigma=0.1)
