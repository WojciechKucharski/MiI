import math
import numpy as np
from typing import List
import matplotlib.pyplot as plt


def m_function(x: float, a: float = 1) -> float:
    if abs(x) > 2:
        return 0
    elif abs(x) < 1:
        return a * x ** 2
    else:
        return 1


def phi_function(k: int, x: float) -> float:
    if k == 0:
        return (2 * math.pi) ** (-0.5)
    else:
        return math.pi ** (-0.5) * math.cos(x * k)


class StaticSystem:
    def __init__(self, m=m_function, phi=phi_function, sigma: float = 1, distr: str = "N"):
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


class MISO:
    def __init__(self, D: int, a: List[float], b: float = 0, sigma_Z: float = 0.2, sigma_X: float = 0.2,
                 mu_X: float = 0, mu_Z: float = 0):
        if len(a) != D:
            raise Exception("Wrong a* vector size")
        self.D, self.a, self.b, self.sigma_Z, self.sigma_X, self.mu_Z, self.mu_X = D, a, b, abs(sigma_Z), abs(
            sigma_X), mu_Z, mu_X

    def simulate(self, N: int, numpy: bool = False, XN: list = [], ZN: list = []):
        N = max(1, N)
        if len(ZN) != N:
            ZN = self.Zn(N)
        if len(XN) != N:
            XN = self.Xn(N)
        YN = []
        for i, Zn in enumerate(ZN):
            Yn = 0
            for j, an in enumerate(self.a):
                Yn += an * XN[i][j]
            YN.append(Yn + Zn)
        if numpy:
            return np.array(XN), np.array(ZN), np.array(YN)
        else:
            return XN, ZN, YN

    def R(self, ZN):
        z = np.array(ZN)
        N = len(ZN)
        R = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if j == i:
                    R[i][j] = (1 + self.sigma_Z) * self.b ** 2
                elif j == i + 1 or j == i - 1:
                    R[i][j] = self.b * self.sigma_Z
        return R

    def cov1(self, N: int):
        XN, ZN, YN = self.simulate(N=N, numpy=True)
        cov = np.linalg.inv(np.matmul(XN.T, XN)) * self.sigma_Z ** 2
        plt.imshow(cov)
        plt.show()

    def cov2(self, N):
        XN, ZN, YN = self.simulate(N=N, numpy=True)
        R = self.R(ZN)
        cov = np.linalg.inv(XN.T @ XN) @ XN.T @ R @ XN @ np.linalg.inv(XN.T @ XN)
        plt.imshow(cov)
        plt.show()

    def Err(self, N: int, L: int):
        Err = 0
        XN = np.array(self.Xn(N))
        for l in range(L):
            XN, _, YN = self.simulate(N=N, numpy=True, XN=XN)
            aN = list(np.matmul(np.linalg.inv(np.array(np.matmul(XN.T, XN))), np.matmul(XN.T, YN)))
            for i, a in enumerate(self.a):
                aN[i] -= a
            Err += float(np.linalg.norm(aN)) ** 2 / L
        return Err

    def ErrInSigma(self, sigma: List[float], N: int, L: int):
        Err = []
        for s in sigma:
            self.sigma_Z = s
            Err.append(self.Err(N, L))
        return Err

    def aN(self, N: int) -> List[float]:
        XN, ZN, YN = self.simulate(N=N, numpy=True)
        return list(np.matmul(np.linalg.inv(np.array(np.matmul(XN.T, XN))), np.matmul(XN.T, YN)))  # abomination

    def Zn(self, N: int) -> List[float]:
        epsilon = list(np.random.normal(self.mu_Z, self.sigma_Z ** 2, N + 1))
        return [epsilon[i + 1] + epsilon[i] * self.b for i in range(N)]

    def Xn(self, N: int) -> List[List[float]]:
        return [list(x) for x in np.random.normal(self.mu_X, self.sigma_X ** 2, (N, self.D))]
