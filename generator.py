import time as t
import math

def random(size = 1, simple = False, use_time = True):
    if use_time:
        x0 = t.time()
    else:
        x0 = math.pi

    if simple == True:
        return r1(correct(size), x0 = x0)
    else:
        return r2(correct(size), c = x0)

def r1(size = 5, x0 = math.pi, z = math.pi**3):
    tab = [x0 % 1]
    for x in range(size + 10):
        tab.append(tab[-1]*z - math.floor(tab[-1]*z))

    if size == 1:
        return tab[-1]
    return tab[-size:]

def r2(size = 5, c = math.pi, k = 10):
    a = r1(k)
    tab = [c % 1]
    for x in range(size + 10):
        new = 0.0
        for i in range(min(x + 1, k)):
            new += tab[-i-1] * a[i]
        tab.append((new+c) % 1)
    if size == 1:
        return tab[-1]
    return tab[-size:]

def correct(x):
    try:
        x = int(abs(x))
    except:
        x = 1
    return x


def rozk1(x):
    y = []
    for i in x:
        y.append(i**0.5)
    return y

def rozk2(x):
    y = []
    for i in x:
        if i<0.5:
            y.append(-1 + (2*i)**0.5)
        else:
            y.append(1 - (2-2*i)**0.5)
    return y

def rozk3(x):
    y = []
    for i in x:
        y.append(-math.log(1-i, math.e))
    return y

def rozk4(x, b = 1, u = 0):
    y = []
    for i in x:
        if i<0.5:
            y.append(u+b*(math.log(2*i, math.e)))
        else:
            y.append(u - b*(math.log(2-2*i, math.e)))
    return y