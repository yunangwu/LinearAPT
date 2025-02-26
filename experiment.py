from algorithm import RandomAlgorithm, LinearAPTAlgorithm, LinearAPT_GP_Algorithm, APTAlgorithm, UCBEAlgorithm, LSEAlgorithm
from sampler import UniformSampler, SoareBAISampler, IrisSampler, WineSampler
from utils import evaluator
from tqdm import tqdm
import json
import numpy as np
import argparse
from utils import algoNames
import os

algoNames = [
    "random",
    "APT",
    "linearAPT",
    "UCBE(-1)",
    "UCBE(0)",
    "UCBE(4)"
    ]

if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler', type=str)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--precision', type=float)
    parser.add_argument('--trial', type=int)
    parser.add_argument('--budget', metavar='N', type=int, nargs='+')

    args = parser.parse_args()

    # define problem
    
    precision = args.precision
    if args.sampler == "uniform_low_dim":
        sampler = UniformSampler(5, 1, 20)
        threshold = args.threshold
    elif args.sampler == "uniform":
        sampler = UniformSampler(20, 1, 20)
        threshold = args.threshold
    elif args.sampler == "iris":
        sampler = IrisSampler(1)
        threshold = sampler.getAverageValue()
    elif args.sampler == "wine":
        sampler = WineSampler(1)
        threshold = sampler.getAverageValue()
    truths = sampler.getAnswer(threshold, precision)

    # experiment settings
    trial = args.trial
    budgetList = args.budget

    algorithms = [
        RandomAlgorithm(),
        APTAlgorithm(),
        LinearAPTAlgorithm(),
        UCBEAlgorithm(-1),
        UCBEAlgorithm(0),
        UCBEAlgorithm(4)
        ]
    

    
    

    trials = {}
    trials["budget"] = budgetList

    for algo, name in zip(algorithms, algoNames):
        mean = []
        for budget in budgetList:
            result = []
            for _ in tqdm(range(trial)):
                algo.solve(sampler, budget, threshold, precision)
                preds = algo.getAnswer(threshold)
                if evaluator(truths, preds):
                    result.append(1)
                else:
                    result.append(0)
            mean.append(np.mean(result))
        trials[name] = mean

    trials["experiment"] = args.sampler
    file_path = f"run/result_{args.sampler}.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(trials, f)
