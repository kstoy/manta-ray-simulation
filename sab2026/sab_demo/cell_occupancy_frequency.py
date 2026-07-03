"""Frequency plot of ball-per-cell occupancy across all sab_demo_convergence runs.

For every (run, cell, timestep) sample with count > 0, tally the occupancy
value (1, 2, 3, ...) and plot how often each value occurs. Zero-occupancy
samples are excluded since they dominate trivially and aren't informative.

Reuses the per-run 'counts' arrays already stored in sab_demo_convergence.pkl,
so no simulation is re-run.

Usage:
    python sab2026/sab_demo/cell_occupancy_frequency.py
"""
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

PKL = Path(__file__).parent / "sab_demo_convergence.pkl"
OUT_PDF = Path(__file__).parent / "cell_occupancy_frequency.pdf"


def main():
    with open(PKL, "rb") as f:
        data = pickle.load(f)

    tally = Counter()
    for res in data["results"]:
        counts = res["counts"]  # (T, ncx, ncy)
        values, freqs = np.unique(counts[counts > 0], return_counts=True)
        for v, f in zip(values.tolist(), freqs.tolist()):
            tally[v] += f

    levels = sorted(tally)
    freqs = [tally[v] for v in levels]

    print("Occupancy value : frequency (cell-timestep samples across all runs)")
    for v, f in zip(levels, freqs):
        print(f"  {v:2d} : {f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(levels, freqs, color="#3477eb", edgecolor="k")
    ax.set_xlabel("Objects in cell")
    ax.set_ylabel("Frequency (cell-timestep samples)")
    ax.set_yscale("log")
    ax.set_xticks(levels)
    ax.set_title(f"Cell occupancy frequency ({data['runs']} runs, zero excluded)")
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"\nPlot saved to {OUT_PDF}")


if __name__ == "__main__":
    main()
