from sampler import BaseSampler
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

class LinearRegressor:
    def __init__(self, dim):
        self.dim = dim
        self.V = np.identity(dim)
        self.b = np.zeros(dim)
        self.theta = None

    def update(self, x, y):
        self.V += np.outer(x, x)
        self.b += y * x
        self.theta = np.linalg.inv(self.V) @ self.b

    def predict(self, x):
        return x @ self.theta
    
class GPRegressor:
    def __init__(self):
        self.X = []
        self.y = []

        self.kernel = DotProduct() + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, random_state=0)

    def update(self, x, y):
        self.X.append(x)
        self.y.append(y)
        self.gp.fit(self.X, self.y)

    def predict(self, x, return_std=False):
        return self.gp.predict(x.reshape(1, -1), return_std=return_std)


from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    def __init__(self):
        self.result = []

    @abstractmethod
    def solve(self, sampler, budget, threshold, precision):
        pass

    def getAnswer(self, threshold):
        return self.result


class RandomAlgorithm(BaseAlgorithm):
    def solve(self, sampler, budget, threshold, precision):
        K = sampler.getArmCount()
        d = sampler.getDim()
        count = np.zeros(K)

        regressor = LinearRegressor(d)

        for t in range(budget):
            if t < K:
                chosenArm = t
            else:
                chosenArm = np.random.randint(0, K)
            reward = sampler.getArmReward(chosenArm)
            vec = sampler.getArmVector(chosenArm)
            regressor.update(vec, reward)
            count[chosenArm] += 1

        self.result = [regressor.predict(sampler.getArmVector(i)) for i in range(K)]
        self.result = [("larger" if r > threshold else "smaller") for r in self.result]


class APTAlgorithm(BaseAlgorithm):
    def solve(self, sampler, budget, threshold, precision):
        K = sampler.getArmCount()
        d = sampler.getDim()
        count = np.zeros(K)
        accum = np.zeros(K)

        for t in range(budget):
            if t < K:
                chosenArm = t
            else:

                def helper(i):
                    left = np.sqrt(count[i])
                    right = accum[i] / count[i]
                    right = np.abs(right - threshold) + precision
                    return left * right

                chosenArm = np.argmin(np.array([helper(i) for i in range(K)]))
            reward = sampler.getArmReward(chosenArm)
            accum[chosenArm] += reward
            count[chosenArm] += 1

        self.result = [accum[i] / count[i] for i in range(K)]
        self.result = [("larger" if r > threshold else "smaller") for r in self.result]


class LinearAPTAlgorithm(BaseAlgorithm):
    def solve(self, sampler, budget, threshold, precision):
        K = sampler.getArmCount()
        d = sampler.getDim()
        count = np.zeros(K)

        regressor = LinearRegressor(d)

        for t in range(budget):
            if t < K:
                chosenArm = t
            else:

                def helper(i):
                    left = np.sqrt(count[i])
                    right = regressor.predict(sampler.getArmVector(i))
                    right = np.abs(right - threshold) + precision
                    return left * right

                chosenArm = np.argmin(np.array([helper(i) for i in range(K)]))
            reward = sampler.getArmReward(chosenArm)
            vec = sampler.getArmVector(chosenArm)
            regressor.update(vec, reward)
            count[chosenArm] += 1

        self.result = [regressor.predict(sampler.getArmVector(i)) for i in range(K)]
        self.result = [("larger" if r > threshold else "smaller") for r in self.result]


class LinearAPT_GP_Algorithm(BaseAlgorithm):
    def solve(self, sampler, budget, threshold, precision):
        K = sampler.getArmCount()
        d = sampler.getDim()
        count = np.zeros(K)

        gp = GPRegressor()

        for t in range(budget):
            if t < K:
                chosenArm = t
            else:

                def helper(i):
                    left = np.sqrt(count[i])
                    right, _ = gp.predict(sampler.getArmVector(i), True)
                    right = np.abs(right - threshold) + precision
                    return left * right

                chosenArm = np.argmin(np.array([helper(i) for i in range(K)]))
            reward = sampler.getArmReward(chosenArm)
            vec = sampler.getArmVector(chosenArm)
            gp.update(vec, reward)
            count[chosenArm] += 1

        self.result = [gp.predict(sampler.getArmVector(i))[0] for i in range(K)]
        self.result = [("larger" if r > threshold else "smaller") for r in self.result]


class UCBEAlgorithm(BaseAlgorithm):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def solve(self, sampler, budget, threshold, precision):
        K = sampler.getArmCount()
        d = sampler.getDim()
        a = (budget - K) / sampler.getComplexity(threshold, precision)
        count = np.zeros(K)

        regressor = LinearRegressor(d)

        for t in range(budget):
            if t < K:
                chosenArm = t
            else:

                def helper(i):
                    mean = regressor.predict(sampler.getArmVector(i))
                    gap = abs(mean - threshold) + precision
                    right = np.sqrt(a / count[i])
                    return gap - right

                chosenArm = np.argmin(np.array([helper(i) for i in range(K)]))
            reward = sampler.getArmReward(chosenArm)
            vec = sampler.getArmVector(chosenArm)
            regressor.update(vec, reward)
            count[chosenArm] += 1

        self.result = [regressor.predict(sampler.getArmVector(i)) for i in range(K)]
        self.result = [("larger" if r > threshold else "smaller") for r in self.result]


class LSEAlgorithm(BaseAlgorithm):
    def __init__(self, beta=1.96):
        super().__init__()
        self.beta = beta

    def solve(self, sampler, budget, threshold, precision, acc=0):
        K = sampler.getArmCount()
        d = sampler.getDim()
        count = np.zeros(K)
        gp = GPRegressor()
        
        H, L, U = set(), set(), set(np.arange(K))
        # C is confidential range, initialized as R
        C = np.zeros((K, 2), dtype=float)
        C[:, 0] = -1e9 * np.ones(K)
        C[:, 1] = 1e9 * np.ones(K)

        for t in range(budget):

            if t < K:
                chosenArm = t
            else:

                max_mu = -1
                chosenArm = -1
                
                for i in U:
                    vec = sampler.getArmVector(i)
                    mu, sigma = gp.predict(vec, True)
                    Q = np.zeros(2)
                    Q[0] = mu - self.beta ** 0.5 * sigma
                    Q[1] = mu + self.beta ** 0.5 * sigma
                    # intersection of Q and C
                    C[i, 0] = max(C[i, 0], Q[0])
                    C[i, 1] = min(C[i, 1], Q[1])
                    if C[i, 0] + acc > threshold:
                        H.add(i)
                    elif C[i, 1] - acc < threshold:
                        L.add(i)

                    if mu > max_mu:
                        max_mu = mu
                        chosenArm = i

                U -= (H | L)
            
            reward = sampler.getArmReward(chosenArm)
            vec = sampler.getArmVector(chosenArm)
            gp.update(vec, reward)
            count[chosenArm] += 1


        #self.result = [
        #    "larger" if i in H else "smaller" for i in range(K)
        #]
        self.result = [gp.predict(sampler.getArmVector(i)) for i in range(K)]
        self.result = [("larger" if r > threshold else "smaller") for r in self.result]