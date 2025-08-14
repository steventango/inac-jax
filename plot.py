from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

colormap = {
    "InAC (PyTorch)": "#4169e1",  # royal blue
    "InAC (JAX)": "#ff4500",  # orange red
    "orchid": "#da8bc3",  # orchid
    "sea green": "#2e8b58",  # sea green
}

evaluation_paths = sorted(
    Path("data").glob("*/output/*/*/0/*_run/evaluations.npy")
)

dfs = []
for path in evaluation_paths:
    evaluations = np.load(path, allow_pickle=True)
    df = pd.DataFrame(
        {
            "x": np.arange(0, len(evaluations) * 10000, 10000) // 10000,
            "y": evaluations * 100,
        }
    )  # Convert to percentage
    backend = path.parts[1]
    task = path.parts[3]
    dataset = path.parts[4]
    df["alg"] = f"InAC ({backend})"
    df["seed"] = path.parent.stem.split("_")[0]
    df["task"] = {
        "Walker2d": "Walker2D",
        "HalfCheetah": "HalfCheetah",
        "Hopper": "Hopper",
        "Ant": "Ant",
    }.get(task, task)
    df["dataset"] = {
        "expert": "Expert",
        "medexp": "M-Expert",
        "medrep": "M-Replay",
        "medium": "Medium",
    }.get(dataset, dataset)
    df["y_rolling"] = df["y"].rolling(window=10, min_periods=1).mean()
    df["alg_seed"] = df["alg"] + " " + df["seed"]
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

new_colormap = {
    f"{k} {seed}": v
    for k, v in colormap.items()
    for seed in df["seed"].unique()
}
colormap.update(new_colormap)

sns.set_style("darkgrid")

tasks = sorted(df["task"].unique())
datasets = sorted(df["dataset"].unique())

fig, axes = plt.subplots(
    len(tasks),
    len(datasets),
    figsize=(4 * len(datasets), 3 * len(tasks)),
    sharex="col",
    sharey="row",
    squeeze=False,
)

for i, task in enumerate(tasks):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]
        df_subset = df[(df["task"] == task) & (df["dataset"] == dataset)]
        if df_subset.empty:
            continue
        sns.lineplot(
            df_subset,
            x="x",
            y="y_rolling",
            hue="alg",
            palette=colormap,
            ax=ax,
            legend=i == len(tasks) - 1 and j == len(datasets) - 1,
        )
        ax.set_title(f"{task} {dataset}")
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel(None)
        ax.set_xlabel(None)

fig.supxlabel("Iteration ($x10^4$)")
fig.supylabel("Normalized Score")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig("img/plot.png", dpi=300)
