import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
from estimators import *
seed = 12343

g = [
    file_generator(), #0
    normal_generator(0,1),#1
    normal_generator(1,1),#2
    normal_generator(0,5),#3
    unit_generator(),#4
    triangle_generator(),#5
    symetric_triangle_generator(),#6
    laplace_generator(),#7
    exp_generator(),#8
    annoying_generator(),#9
    cauchy_generator(0,1)#10
]


