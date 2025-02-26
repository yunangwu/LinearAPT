import numpy as np
import tqdm as tqdm


def evaluator(truths, preds):
    def helper(truth, pred):
        if truth == "irrelevant":
            return True
        return truth == pred

    result = [helper(t, p) for t, p in zip(truths, preds)]
    return all(result)


algoNames = [
        "linearAPT_GP",
        "LSE",
        "random",
        "APT",
        "linearAPT",
        "UCBE(-1)",
        "UCBE(0)",
        "UCBE(4)"
        ]