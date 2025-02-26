import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
from experiment import algoNames
import os

if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    args = parser.parse_args()

    file_path = args.file_path

    with open(file_path, 'r') as f:
        trials = json.load(f)

    font = {"size": 16}
    linewidth = 2.5
    matplotlib.rc("font", **font)

    plt.figure(figsize=(9, 6))

    markers = ["s","s","s","s","s","s"]
    linestyles = ["solid","dashed","dashed","dotted","dotted","dotted"]
    colors = ["cyan","black","purple","blue","green","red"]


    for algoname, marker, linestyle, color in zip(algoNames, markers, linestyles, colors):
    
        plt.plot(
            trials["budget"],
            [np.log(1 - x) for x in trials[algoname]],
            label=algoname,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
        )

    plt.xlabel("Budget")
    plt.ylabel("log(probability of faliure)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"run/plot_{trials['experiment']}.png")