import pickle
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "benchmark_results.pkl"


if __name__ == "__main__":
    with open(RESULTS_PATH, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    ball_counts = data["ball_counts"]
    timesteps = data.get("timesteps", "?")
    repetitions = data.get("repetitions", "?")

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, (means, stds) in results.items():
        ax.plot(ball_counts, means, marker="o", label=label)
        ax.fill_between(ball_counts, means - stds, means + stds, alpha=0.2)

    ax.set_xlabel("Number of objects")
    ax.set_ylabel("Time (seconds)")
    ax.legend(title="Surface size (# rods)")
    ax.set_xticks(ball_counts)
    fig.tight_layout()

    out = Path(__file__).parent / "benchmark.png"
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.show()
