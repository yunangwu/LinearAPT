import numpy as np
from sklearn.datasets import load_iris, load_wine


class BaseSampler:
    def __init__(self, dim, noiseStd):
        self.dim = dim
        self.noiseStd = noiseStd
        self.parameter = None
        self.armVector = None
        self.armCount = None

    def getArmCount(self):
        return self.armCount

    def getDim(self):
        return self.dim

    def getArmVector(self, i):
        return self.armVector[i]

    def _getTrueReward(self, i):
        return self.armVector[i] @ self.parameter

    def getArmReward(self, i):
        noise = np.random.normal(scale=self.noiseStd)
        return self._getTrueReward(i) + noise

    def getComplexity(self, threshold, precision):
        deltas = [
            abs(self._getTrueReward(i) - threshold) / precision
            for i in range(self.getArmCount())
        ]
        return np.sum([val ** -2 for val in deltas])

    def getAnswer(self, threshold, prceision):
        rewards = np.array([self._getTrueReward(i) for i in range(self.armCount)])

        def condition(x):
            diff = x - threshold
            if abs(diff) < prceision:
                return "irrelevant"
            elif diff > 0:
                return "larger"
            else:
                return "smaller"

        return [condition(r) for r in rewards]


class SoareBAISampler(BaseSampler):
    def __init__(self, dim, noiseStd, omega):
        super().__init__(dim, noiseStd)
        self.omega = omega
        self.armCount = dim + 1
        self.parameter = self._initParameter()
        self.armVector = self._initArmVector()

    def _initParameter(self):
        params = np.zeros(self.dim)
        params[0] = 2
        return params

    def _initArmVector(self):
        I = np.identity(self.dim)
        column = np.zeros(self.dim)
        column[0], column[1] = np.cos(self.omega), np.sin(self.omega)
        return np.vstack((I, column))


class UniformSampler(BaseSampler):
    def __init__(self, dim, noiseStd, armCount):
        super().__init__(dim, noiseStd)
        self.armCount = armCount
        self.parameter = self._initParameter()
        self.armVector = self._initArmVector()

    def _sample(self):
        """
        sample from [-1, 1]^d
        """
        return np.random.uniform(size=self.dim) * 2 - 1

    def _initParameter(self):
        return self._sample()

    def _initArmVector(self):
        return np.array([self._sample() for _ in range(self.armCount)])


class IrisSampler(BaseSampler):
    def __init__(self, noiseStd):
        super().__init__(4, noiseStd)
        self.armCount = 150
        self.parameter = self._initParameter()
        self.armVector = self._initArmVector()

    def _sample(self):
        """
        sample from [-1, 1]^d
        """
        return np.random.uniform(size=self.dim) * 2 - 1

    def _initParameter(self):
        return self._sample()

    def _initArmVector(self):
        return load_iris().data

    def getAverageValue(self):
        vals = self.armVector @ self.parameter
        return np.mean(vals)


class WineSampler(BaseSampler):
    def __init__(self, noiseStd):
        super().__init__(13, noiseStd)
        self.armCount = 178
        self.parameter = self._initParameter()
        self.armVector = self._initArmVector()

    def _sample(self):
        """
        sample from [-1, 1]^d
        """
        return np.random.uniform(size=self.dim) * 2 - 1

    def _initParameter(self):
        return self._sample()

    def _initArmVector(self):
        return load_wine().data

    def getAverageValue(self):
        vals = self.armVector @ self.parameter
        return np.mean(vals)
