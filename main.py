from generator import random
import matplotlib.pyplot as plt
from functions import *

plt.hist(random(10**5, True))
plt.show()