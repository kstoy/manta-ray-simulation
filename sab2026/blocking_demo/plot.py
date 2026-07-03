"""Plot blocking and nonblocking results together: pairwise and center distance over time."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
SERIES = [
    ("blocking",     HERE / "data_blocking.pkl",    "C0"),
    ("none-blocking", HERE / "data_nonblocking.pkl", "C1"),
]
OUTPUT_PATH = HERE / "plot.png"


def mean_std_band(ax, t, runs, color, label):
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    ax.plot(t, mean, color=color, label=label)
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)


if __name__ == "__main__":
    fig, (ax_pair, ax_center) = plt.subplots(1, 2, figsize=(12, 5))

    n_runs = None
    T = None
    for label, path, color in SERIES:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if T is None:
            T = data["timesteps"]
            n_runs = data["n_runs"]
        t = np.arange(data["timesteps"]) * 0.1
        mean_std_band(ax_pair,   t, data["pair_dist"],   color, label)
        mean_std_band(ax_center, t, data["center_dist"], color, label)

    ax_pair.set_xlabel("Time (s)")
    ax_pair.set_ylabel("Average pairwise distance")
    ax_pair.set_title("Average distance between objects")
    ax_pair.legend()

    ax_center.set_xlabel("Time (s)")
    ax_center.set_ylabel("Average distance to center")
    ax_center.set_title("Average distance to surface center")
    ax_center.legend()

    # Equalize y-axis span so both panels have the same units per tick.
    y0, y1 = ax_pair.get_ylim()
    span = y1 - y0
    c0, c1 = ax_center.get_ylim()
    cy = (c0 + c1) / 2
    ax_center.set_ylim(cy - span / 2, cy + span / 2)

    fig.tight_layout()

    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH}")
    plt.show()
