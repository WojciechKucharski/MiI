import time as t
import math

def random(size = 1, simple = False):
    if simple == True:
        return r1(correct(size))
    else:
        return r2(correct(size))

def correct(x):
    try:
        x = int(abs(x))
    except:
        x = 1
    return x

def r1(size = 5):
    z = 123.32
    tab = [t.time() % 1]
    for _ in range(size + 10):
        tab.append(tab[-1]*z - math.floor(tab[-1]*z))
    if size == 1:
        return tab[-1]
    return tab[-size:]

def r2(size = 5):
    c = math.pi
    k = 10
    a = r1(k)
    tab = [t.time() % 1]
    for x in range(size + 10):
        new = 0.0
        for i in range(min(x + 1, k)):
            new += tab[-i-1] * a[i]
        new = (new+c) % 1
        tab.append(new)
    if size == 1:
        return tab[-1]
    return tab[-size:]