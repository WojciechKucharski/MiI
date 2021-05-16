import math
import numpy as np
from typing import List


class StaticSystem:
    def __init__(self, sigma: float = 1, distr: str = "N"):
        self.sigma, self.distr = abs(sigma), distr

    @staticmethod
    def m(x: float) -> float:
        x+=1
        if x > 2 or x < -2:
            return 0
        if 2 >= x > 1 or -2 <= x < -1:
            return 1
        else:
            return x ** 2

    @staticmethod
    def fi(k: int, x: float) -> float:
        if k == 0:
            return (2 * math.pi) ** (-1 / 2)
        else:
            return math.cos(k * x) * math.pi ** (-1 / 2)

    def simulate(self, N):
        Xn = list(np.random.uniform(-math.pi, math.pi, N))
        if self.distr == "N":
            Zn = list(np.random.normal(0, self.sigma, N))
        elif self.distr == "C":
            Zn = [self.sigma * math.tan(math.pi * (x - 0.5)) for x in np.random.uniform(0, 1, N)]
        else:
            raise Exception(f"{self.distr} not supported")
        Yn = [Zn[i] + self.m(Xn[i]) for i in range(N)]
        return Xn, Zn, Yn


    def alfa_k(self, Xn: List[float], Yn: List[float], k: int) -> float:
        return float(np.mean(
            [Yn[i] * self.fi(k=k, x=x) for i, x in enumerate(Xn)]
        ))

    def beta_k(self, Xn: List[float], k: int) -> float:
        return float(np.mean(
            [self.fi(k=k, x=x) for x in Xn]
        ))

    def mN(self, N: int, L: int, X: List[float]) -> List[float]:
        Xn, Zn, Yn = self.simulate(N)
        mN = []
        for x in X:
            gN = sum([
                self.alfa_k(Xn, Yn, k) * self.fi(k=k, x=x) for k in range(L + 1)])
            fN = sum([
                self.beta_k(Xn, k) * self.fi(k=k, x=x) for k in range(L + 1)])
            if fN == 0:
                mN.append(0)
            else:
                mN.append(gN / fN)
        return mN

    def valid(self, N: int, L: List[int], Q: int) -> List[float]:
        qVector = [2 * q / Q for q in list(np.linspace(-Q, Q, 2 * Q + 1))]
        output = []
        for l in L:
            mN = self.mN(N=N, L=l, X=qVector)
            output.append(sum([(mN[i] - self.m(q)) ** 2 for i, q in enumerate(qVector)]) / (2 * Q))
        return output
