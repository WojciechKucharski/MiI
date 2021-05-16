import math
import numpy as np
from typing import List

class StaticSystem():
    def __init__(self, sigma: float = 1, a: float = 1):
        self.sigma = abs(sigma)
        self.a = a

    def m(self, x: float) -> float:
        if x > 2 or x < -2:
            return 0
        if x <= 2 and x > 1 or x >= -2 and x < -1:
            return 1
        else:
            return self.a * x ** 2


    def simulate(self, N):
        Xn = list(np.random.uniform(-math.pi, math.pi, N))
        Zn = list(np.random.normal(0, self.sigma, N))
        Yn = [Zn[i] + self.m(Xn[i]) for i in range(N)]
        return Xn, Zn, Yn


    def fi(self, k: int, x: float) -> float:
        if k == 0:
            return (2 * math.pi) ** (-1/2)
        else:
            return math.cos(k * x) * math.pi ** (-1/2)

    def alfa_k(self,Xn: List[float], Yn: List[float], k: int) -> float:
        return float(np.mean(
            [Yn[i] * self.fi(k=k, x=x) for i, x in enumerate(Xn)]
        ))

    def beta_k(self, Xn: List[float], k: int) -> float:
        return float(np.mean(
            [self.fi(k=k, x=x) for x in Xn]
        ))

    def mN(self, N: int, L: int, X: List[float]):
        Xn, Zn, Yn = self.simulate(N)
        mN = []

        for x in X:
            gN = sum([
                self.alfa_k(Xn, Yn, k) * self.fi(k=k, x=x) for k in range(L+1)
            ])
            fN = sum([
                self.beta_k(Xn, k) * self.fi(k=k, x=x) for k in range(L + 1)
            ])
            if fN == 0:
                mN.append(0)
            else:
                mN.append(gN/fN)

        return mN
