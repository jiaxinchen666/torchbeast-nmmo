import pandas as pd
from matplotlib import pyplot as plt
import argparse

STATS = [
    "total_loss", "mean_episode_return", "mean_episode_step", "pg_loss",
    "baseline_loss", "entropy_loss"
]
N_COLUMN = 3


def plot(log_file):
    fig, axes = plt.subplots(len(STATS) // N_COLUMN, N_COLUMN)
    log_df = pd.read_csv(log_file, sep=",")
    for i, key in enumerate(STATS):
        assert "step" in log_df.columns and key in log_df.columns
        df = log_df[["step", key]].dropna()
        ax = axes[i // N_COLUMN, i % N_COLUMN]
        ax.set_title(key)
        ax.plot(df["step"], df[key])
    plt.savefig("results/logs.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile",
        type=str,
        default="results/logs.csv",
    )
    args = parser.parse_args()
    plot(args.logfile)
