import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def plot_curves(exp_path):
    with open(os.path.join(exp_path, "losses.pkl"), "rb") as f:
        losses = pickle.load(f)

    evaluations = np.load(os.path.join(exp_path, "evaluations.npy"))

    for k, v in losses.items():
        plt.figure()
        plt.plot(v)
        plt.title(k)
        plt.savefig(os.path.join(exp_path, f"{k}.png"))
        plt.close()

    plt.figure()
    plt.plot(evaluations)
    plt.title("evaluations")
    plt.savefig(os.path.join(exp_path, "evaluations.png"))
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default="/workspaces/inac_pytorch/data/JAX/output/Hopper/medrep/0/0_run")
    args = parser.parse_args()
    plot_curves(args.exp_path)
