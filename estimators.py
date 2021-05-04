import math
import numpy as np
from typing import List


def evaluate(fun: str, x: float) -> float:
    return eval(fun)


class generator2:
    def __init__(self, PDF: str, CDF: str, Quantile: str, mu: float = 0, sigma: float = 1):
        self.PDF, self.CDF, self.Quantile = PDF, CDF, Quantile
        self.mu, self.sigma = mu, sigma

    def rand(self, size: int) -> List[float]:
        if self.Quantile == "normal":
            return list(np.random.normal(self.mu, self.sigma, size))
        return [evaluate(self.Quantile, x) for x in list(np.random.rand(size))]


class generator_file:
    def __init__(self, path: str = "dataLab4.txt"):
        self.data = [float(x) for x in open(path, "r")]
        self.maxN = len(self.data)
        self.PDF, self.CDF = None, None

    def rand(self, size: int) -> List[float]:
        output = []
        for i in range(size // self.maxN):
            output += self.data
        output += self.data[:size % self.maxN]
        return output


class Tester:
    def __init__(self, generator):
        self.g = generator
        self.rand(5)

    def rand(self, size: int) -> List[float]:
        return self.g.rand(size)

    def CDF(self, x: float) -> float:
        return evaluate(self.g.CDF, x)

    def PDF(self, x: float) -> float:
        return evaluate(self.g.PDF, x)

    def estimator(self, parameter: str, N: List[int], L: int = 1) -> List[float]:
        if parameter not in ["u", "s", "S"]:
            raise Exception(f"Estimator of parameter {parameter} is not supported")
        elif parameter == "u":
            return [float(np.mean([float(np.mean(self.rand(i))) for _ in range(L)])) for i in N]
        elif parameter == "s":
            return [float(np.mean([float(np.var(self.rand(i))) for _ in range(L)])) for i in N]
        elif parameter == "S":
            return [float(np.mean([i / (i + 1) * float(np.var(self.rand(i))) for _ in range(L)])) for i in N]
        raise Exception("Something went wrong while estimating parameter")

    def Err(self, parameter: str, value: float, N: List[int], L: int = 1) -> List[float]:
        if parameter not in ["u", "s", "S"]:
            raise Exception(f"Estimator of parameter {parameter} is not supported")
        elif parameter == "u":
            return [float(np.mean([(value - float(np.mean(self.rand(i)))) ** 2 for _ in range(L)])) for i in N]
        elif parameter == "s":
            return [float(np.mean([(value - float(np.var(self.rand(i)))) ** 2 for _ in range(L)])) for i in N]
        elif parameter == "S":
            return [float(np.mean([(value - i / (i + 1) * float(np.var(self.rand(i)))) ** 2 for _ in range(L)])) for i
                    in N]
        raise Exception("Something went wrong while estimating parameter")

    def FN(self, N: int, X: List[float]) -> List[float]:
        data = self.rand(N)
        return [float(np.mean([float(Xn <= x) for Xn in data])) for x in X]

    def fN(self, N: int, X: List[float], kernel: str = "float(x <= 0.5 and x >= -0.5)", hN: float = 1) -> List[float]:
        data = self.rand(N)
        return [1 / hN * float(np.mean([evaluate(kernel, (Xn - x) / hN) for Xn in data])) for x in X]

    def DN(self, N: int, X: List[float]) -> float:
        data = self.rand(N)
        return max([self.CDF(x) - float(np.mean([float(Xn <= x) for Xn in data])) for x in X])

    def ErrFN(self, N: int, X: List[float], L: int = 1) -> float:
        output = 0
        for _ in range(L):
            data = self.rand(N)
            output += float(
                np.mean([(self.CDF(x) - float(np.mean([float(Xn <= x) for Xn in data]))) ** 2 for x in X])) / L
        return output

    def ErrfN(self, N: int, X: List[float], L: int = 1, kernel: str = "float(x <= 0.5 and x >= -0.5)",
              hN: float = 1) -> float:
        output = 0
        for _ in range(L):
            data = self.rand(N)
            output += float(np.mean(
                [(self.PDF(x) - 1 / hN * float(np.mean([evaluate(kernel, (Xn - x) / hN) for Xn in data]))) ** 2 for x in
                 X])) / L
        return output

    def varFN(self, N: int, X: List[float], L: int = 1) -> List[float]:
        output = [0] * len(X)
        for _ in range(L):
            data = self.rand(N)
            new = [(self.CDF(x) - float(np.mean([float(Xn <= x) for Xn in data]))) ** 2 / L for x in X]
            for i in range(len(X)):
                output[i] += new[i]
        return output


def cauchy_generator(mu: float, sigma: float):
    return generator2(
        PDF=f"1 / (math.pi * {sigma} (1 + ((x - {mu}) / {sigma}) ** 2))",
        CDF=f"1 / math.pi * math.atan((x - {mu}) / {sigma}) + 0.5",
        Quantile=f"{mu} + {sigma} * math.tan(math.pi * (x - 0.5))"
    )


def normal_generator(mu: float, sigma: float):
    return generator2(
        PDF=f"1 / ({sigma} * math.sqrt(2*math.pi)) * math.exp(-0.5 * ((x - {mu}) / {sigma}) ** 2)",
        CDF=f"0.5 * (1 + math.erf((x - {mu}) / ({sigma} * math.sqrt(2))))",
        Quantile="normal",
        mu=mu, sigma=sigma
    )


unit_generator = generator2(PDF="float(x<1 and x>0)", CDF="float(x<1 and x>0) * x + float(x>=1)", Quantile="x")
triangle_generator = generator2(PDF="float(float(x<1 and x>0) * x * 2)", CDF="float(x<1 and x>0) * x ** 2",
                                Quantile="math.sqrt(x)")
