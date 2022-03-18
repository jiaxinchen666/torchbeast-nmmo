import pandas as pd
from matplotlib import pyplot as plt

COLUMNS = ["step", "mean_episode_return"]


def plot(log_file):
    df = pd.read_csv(log_file, sep=",")
    df = df[COLUMNS].dropna()
    x_label, y_label = COLUMNS
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(df[x_label], df[y_label])
    plt.savefig("test_pong.png")


if __name__ == "__main__":
    log_file = "./results/logs.csv"
    plot(log_file)
