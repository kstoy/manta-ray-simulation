"""Quick top-down scatter of ball positions to check letter legibility.

Usage:
    python scripts/topdown.py output/sab_test.pkl            # last frame
    python scripts/topdown.py output/sab_test.pkl --frame -1 --out output/td.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import simdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--out", default="output/topdown.png")
    args = ap.parse_args()

    data = simdata.load(args.input)
    balls = data["ballsstates"]
    radii = data["ballsradiuses"]
    config = data["config"]
    rods = data["rodsstates"][args.frame]

    frame = balls[args.frame]
    x, y = frame[:, 0], frame[:, 1]

    gx = (config.GRIDSIZEX - 1) * config.D_RODS
    gy = (config.GRIDSIZEY - 1) * config.D_RODS

    fig, ax = plt.subplots(figsize=(gx / gy * 8 + 1, 8))
    # rod grid as faint dots
    rx = rods[:, :, 0].ravel()
    ry = rods[:, :, 1].ravel()
    ax.scatter(rx, ry, s=4, c="lightgray", zorder=1)
    # balls
    ax.scatter(x, y, s=300, c="tab:blue", edgecolors="k", zorder=2)
    ax.set_xlim(-0.3, gx + 0.3)
    ax.set_ylim(-0.3, gy + 0.3)
    ax.set_aspect("equal")
    ax.set_title(f"{args.input} frame {args.frame}  ({len(x)} balls)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
