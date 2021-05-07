import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from estimators import *
gen = laplace_generator()

x = np.linspace(-3,3,1000)

plt.plot(
    x, evaluate(kernels["gauss"], x),
x, evaluate(kernels["triangle"], x),
x, evaluate(kernels["rectangle"], x),
x, evaluate(kernels["tricube"], x),
x, evaluate(kernels["epechnikov"], x),
)
plt.legend(["Gauss", "Triangle", "Rectangle", "Tricube", "Epochnikov"])
plt.show()
print(kernels[1])