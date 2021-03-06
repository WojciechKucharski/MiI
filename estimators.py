import math
import numpy as np
from typing import List
from functools import cache

def evaluate(fun: str, x):
    if type(x) != list and type(x) != type(np.linspace(0, 1, 3)):
        return evaluate2(fun, x)
    else:
        return [evaluate2(fun, xn) for xn in x]

@cache
def evaluate2(fun: str, x) -> float:
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
    def __init__(self, path: str):
        self.data = [float(x) for x in open(path, "r")]
        self.maxN = len(self.data)
        self.PDF_i = None
        self.CDF_i = None

    def rand(self, size: int) -> List[float]:
        output = []
        for i in range(size // self.maxN):
            output += self.data
        output += self.data[:size % self.maxN]
        return output

    @property
    def PDF(self):
        if self.PDF_i == None:
            raise Exception("PDF is not included in the file")
        else:
            return self.PDF_i

    @property
    def CDF(self):
        if self.CDF_i == None:
            raise Exception("CDF is not included in the file")
        return self.CDF_i


def mean_estimator(data_in, med):
    if med:
        return float(np.median(data_in))
    else:
        return float(np.mean(data_in))
def var_estimator(data_in, biased, med):
    mu = mean_estimator(data_in, med)
    N = len(data_in)

    if med:
        return float(np.median([(mu - x) ** 2 for x in data_in]))
    else:
        return 1 / (N - float(biased)) * float(sum([(mu - x) ** 2 for x in data_in]))

class Tester:
    def __init__(self, generator):
        self.g = generator
        self.rand(5)

    def rand(self, size: int) -> List[float]:
        #np.random.seed(8888)
        return self.g.rand(size)

    def PDF(self, x):
        return evaluate(self.g.PDF, x)

    def CDF(self, x):
        return evaluate(self.g.CDF, x)

    def estimator(self, parameter: str, N: List[int], L: int = 1, med: bool = False) -> List[float]:
        if parameter not in ["u", "s", "S"]:
            raise Exception(f"Estimator of parameter {parameter} is not supported")
        elif parameter == "u":
            return [float(np.mean([mean_estimator(self.rand(i), med) for _ in range(L)])) for i in N]
        elif parameter == "s":
            return [float(np.mean([var_estimator(self.rand(i), False, med) for _ in range(L)])) for i in N]
        elif parameter == "S":
            return [float(np.mean([var_estimator(self.rand(i), True, med) for _ in range(L)])) for i in N]
        raise Exception("Something went wrong while estimating parameter")


    def Err(self, parameter: str, value: float, N: List[int], L: int = 1, med: bool = False) -> List[float]:
        if parameter not in ["u", "s", "S"]:
            raise Exception(f"Estimator of parameter {parameter} is not supported")
        elif parameter == "u":
            return [float(np.mean([(value - mean_estimator(self.rand(i), med)) ** 2 for _ in range(L)])) for i in N]
        elif parameter == "s":
            return [float(np.mean([(value - var_estimator(self.rand(i), False, med)) ** 2 for _ in range(L)])) for i in N]
        elif parameter == "S":
            return [float(np.mean([(value - var_estimator(self.rand(i), True, med)) ** 2 for _ in range(L)])) for i
                    in N]
        raise Exception("Something went wrong while estimating parameter")

    def FN(self, N: int, X: List[float]) -> List[float]:
        data = self.rand(N)
        return [float(np.mean([float(Xn <= x) for Xn in data])) for x in X]

    def fN(self, N: int, X: List[float], kernel: str = "float(x <= 0.5 and x >= -0.5)", hN: float = 1) -> List[float]:
        data = self.rand(N)
        return [1 / hN * float(np.mean([evaluate2(kernel, (Xn - x) / hN) for Xn in data])) for x in X]

    def DN(self, N: List[int], X: List[float]) -> List[float]:
        output = []
        for n in N:
            data = self.rand(n)
            output.append(max([abs(self.CDF(x) - float(np.mean([float(Xn <= x) for Xn in data]))) for x in X]))
        return output

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
                [(self.PDF(x) - 1 / hN * float(np.mean([evaluate2(kernel, (Xn - x) / hN) for Xn in data]))) ** 2 for x in
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
    return Tester(generator2(
        PDF=f"1 / (math.pi * {sigma} * (1 + ((x - {mu}) / {sigma}) ** 2))",
        CDF=f"1 / math.pi * math.atan((x - {mu}) / {sigma}) + 0.5",
        Quantile=f"{mu} + {sigma} * math.tan(math.pi * (x - 0.5))"
    ))

def symetric_triangle_generator():
    return Tester((generator2(
        PDF="(1 - abs(x)) * float(x<=1 and x>=-1)",
        CDF="float(x>1) + (0.5*x**2+0.5+x)*float(x>=-1 and x<0) + (-0.5*x**2+x+0.5)*float(x>=0 and x<=1)",
        Quantile="float(x > 0.5) * ((2 * x) ** 0.5 - 1) + float(x <= 0.5) * (1 - (2 - 2 * x) ** 0.5)"
    )))

def exp_generator():
    return Tester((generator2(
        PDF="math.exp(-x) * float(x>=0)",
        CDF="(1-math.exp(-x)) * float(x>=0)",
        Quantile="-math.log(1-x, math.exp(1))"
    )))

def laplace_generator():
    return Tester((generator2(
        PDF="0.5*math.exp(x) * float(x<0) + 0.5*math.exp(-x) * float(x>=0) ",
        CDF="0.5*math.exp(x) * float(x<0) + (1-0.5*math.exp(-x)) * float(x>=0)",
        Quantile="math.log(2*x,math.exp(1)) * float(x<0.5)-math.log(2-2*x,math.exp(1)) * float(x>=0.5)"
    )))

def normal_generator(mu: float, sigma: float):
    return Tester(generator2(
        PDF=f"1 / ({sigma} * math.sqrt(2*math.pi)) * math.exp(-0.5 * ((x - {mu}) / {sigma}) ** 2)",
        CDF=f"0.5 * (1 + math.erf((x - {mu}) / ({sigma} * math.sqrt(2))))",
        Quantile="normal",
        mu=mu, sigma=sigma
    ))

def annoying_generator():
    return Tester(generator2(
        PDF="50 * float(x>=0 and x<0.01) + 50/99 * float(x>=0.01 and x<=1)",
        CDF="x*50 * float(x>=0 and x<0.01) + (x * 50/99 + 49/99) * float(x>=0.01 and x<=1) + float(x>1)",
        Quantile="x/50*float(x<0.5) + (x * 99/50 - 49/50)*float(x>=0.5)"
    ))
def file_generator(path: str = "dataLab4.txt"):
    return Tester(generator_file(path))

def unit_generator():
    return Tester(generator2(PDF="float(x<1 and x>0)", CDF="float(x<1 and x>0) * x + float(x>=1)", Quantile="x"))

def triangle_generator():
    return Tester(generator2(PDF="float(float(x<1 and x>0) * x * 2)", CDF="float(x<1 and x>0) * x ** 2 + float(x>1)",
                                Quantile="math.sqrt(x)"))


class staticSystem:
    def __init__(self, Zn=normal_generator(0, 0.2), m: str = "math.atan(x)", a: float = 1, mu: float = 0,
                 sigma: float = 1,
                 U: List[float] = (-2, 2)):
        self.m, self.a, self.mu, self.sigma, self.U = m, a, mu, sigma, U
        self.Zn = Zn

    def m_fun(self, x):
        return evaluate(self.m, x * self.a)

    def simulate(self, size: int):
        Xn = [x * (self.U[1] - self.U[0]) + self.U[0] for x in list(np.random.rand(size))]
        Zn = self.Zn.rand(size)
        Yn = [Zn[i] + evaluate2(self.m, Xn[i] * self.a) for i in range(size)]
        return Xn, Zn, Yn

    def mN(self, N: int, X: List[float], hN: float = 1, kernel: str = "float(x <= 0.5 and x >= -0.5)", simulated = None) -> List[float]:
        if simulated == None:
            Xn, Zn, Yn = self.simulate(N)
        else:
            Xn, Zn, Yn = simulated[0], simulated[1], simulated[2]

        return [
            
            sum([Yn[i] * evaluate2(kernel, (Xn[i] - x) / hN) for i in range(N)]) 
            / 
                
                max(sum([evaluate2(kernel, (Xn[i] - x) / hN) for i in range(N)]), 0.000001) 
                
                for x in X]



    def valid(self, h: List[float], N: int, Q: int = 100, kernel: str = "float(x <= 0.5 and x >= -0.5)") -> List[float]:
        q = [x / Q for x in np.linspace(-Q, Q, Q * 2)]
        m = [evaluate2(self.m, x * self.a) for x in q]
        Xn, Zn, Yn = self.simulate(N)
        return [1 / (2 * Q) * sum([
            (m[i] - mN) ** 2 for i, mN in enumerate(self.mN(N=N, X=q, hN=hN, kernel=kernel, simulated=(Xn, Zn, Yn)))
        ]) for hN in h]


kernels = {
    "gauss":"(2*math.pi)**(-1/2)*math.exp(-x**2/2)",
    "tricube":"float(70/81*(1-abs(x)**3)**3) * float(x<=1 and x>=-1)",
    "rectangle":"float(x>=-0.5 and x<=0.5)",
    "triangle":"(1-abs(x)) * float(x<=1 and x>=-1)",
    "epechnikov":"float(3/4*(1-x**2)) * float(x<=1 and x>=-1)",
    "g":"(2*math.pi)**(-1/2)*math.exp(-x**2/2)",
    "tc":"float(70/81*(1-abs(x)**3)**3) * float(x<=1 and x>=-1)",
    "r":"float(x>=-0.5 and x<=0.5)",
    "t":"(1-abs(x)) * float(x<=1 and x>=-1)",
    "e":"float(3/4*(1-x**2)) * float(x<=1 and x>=-1)",
    3:"(2*math.pi)**(-1/2)*math.exp(-x**2/2)",
    4:"float(70/81*(1-abs(x)**3)**3) * float(x<=1 and x>=-1)",
    1:"float(x>=-0.5 and x<=0.5)",
    2:"(1-abs(x)) * float(x<=1 and x>=-1)",
    5:"float(3/4*(1-x**2)) * float(x<=1 and x>=-1)",
    "Gauss":"(2*math.pi)**(-1/2)*math.exp(-x**2/2)",
    "Tricube":"float(70/81*(1-abs(x)**3)**3) * float(x<=1 and x>=-1)",
    "Prostok??t":"float(x>=-0.5 and x<=0.5)",
    "Tr??jk??t":"(1-abs(x)) * float(x<=1 and x>=-1)",
    "Epanechnikow":"float(3/4*(1-x**2)) * float(x<=1 and x>=-1)"
}

def inRange(a, b, c):
    return min(c, max(a, b))