import math
import numpy as np
from typing import List


class StaticSystem:
    def __init__(self, m, phi, sigma: float = 1, distr: str = "N"):
        self.sigma, self.distr = abs(sigma), distr
        self.m, self.phi = m, phi


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
            [Yn[i] * self.phi(k=k, x=x) for i, x in enumerate(Xn)]
        ))

    def beta_k(self, Xn: List[float], k: int) -> float:
        return float(np.mean(
            [self.phi(k=k, x=x) for x in Xn]
        ))

    def mN(self, N: int, L: int, X: List[float]) -> List[float]:
        Xn, Zn, Yn = self.simulate(N)
        mN = []
        for x in X:
            gN = sum([
                self.alfa_k(Xn, Yn, k) * self.phi(k=k, x=x) for k in range(L + 1)])
            fN = sum([
                self.beta_k(Xn, k) * self.phi(k=k, x=x) for k in range(L + 1)])
            if fN == 0:
                mN.append(0)
            else:
                mN.append(gN / fN)
        return mN

    def plot_alfa_k(self, N: int, L: List[int]) -> List[float]:
        Xn, Zn, Yn = self.simulate(N)
        return [self.alfa_k(Xn, Yn, k) for k in L]

    def valid(self, N: int, L: List[int], Q: int) -> List[float]:
        qVector = [2 * q / Q for q in list(np.linspace(-Q, Q, 2 * Q + 1))]
        output = []
        for l in L:
            mN = self.mN(N=N, L=l, X=qVector)
            output.append(sum([(mN[i] - self.m(q)) ** 2 for i, q in enumerate(qVector)]) / (2 * Q))
        return output
