"""Pattern coverage of a simulation as a function of time.

Computes, for every recorded frame, the fraction of the target pattern's cells
that have a ball sitting on them, and writes it to a two-column text file:

    # time_s  coverage
    0.000     0.0143
    0.100     0.0286
    ...

Coverage = (# pattern cells containing >=1 ball) / (# pattern cells).  Each ball
is binned to the cell it occupies, floor(x/D), floor(y/D) -- the same notion of
occupancy the controller uses -- so a single ball cannot count for more than the
one cell it is in.

Usage:
    python scripts/coverage.py output/mozart_stochastic.pkl
    python scripts/coverage.py output/mozart_stochastic.pkl \
        --experiment experiments/mozart_stochastic.py --out output/coverage.txt
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import simdata


def _load_bitmap(experiment_path):
    spec = importlib.util.spec_from_file_location("_experiment", experiment_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bit = np.asarray(module.BIT_MAP).astype(str)
    return bit == '1'


def coverage_over_time(ballsstates, pattern, D):
    """Per-frame fraction of pattern cells that contain at least one ball.

    A ball at world (x, y) occupies cell (floor(x/D), floor(y/D)).
    """
    ny, nx = pattern.shape
    n_cells = int(pattern.sum())
    cov = np.empty(len(ballsstates))
    for t, frame in enumerate(ballsstates):
        f = np.asarray(frame)
        cj = np.floor(f[:, 0] / D).astype(int)   # cx
        ci = np.floor(f[:, 1] / D).astype(int)   # cy
        valid = (ci >= 0) & (ci < ny) & (cj >= 0) & (cj < nx)
        occ = np.zeros((ny, nx), dtype=bool)
        occ[ci[valid], cj[valid]] = True
        cov[t] = int(occ[pattern].sum()) / n_cells
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="simulation .pkl file")
    ap.add_argument("--experiment", default="experiments/mozart_stochastic.py",
                    help="experiment file holding BIT_MAP (default: mozart_stochastic)")
    ap.add_argument("--out", default="output/coverage.txt")
    args = ap.parse_args()

    data = simdata.load(args.input)
    ballsstates = data["ballsstates"]
    config = data["config"]

    pattern = _load_bitmap(args.experiment)        # [cy, cx]
    cov = coverage_over_time(ballsstates, pattern, config.D_RODS)
    times = np.arange(len(cov)) * config.DT

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# time_s  coverage\n")
        for t, c in zip(times, cov):
            f.write(f"{t:.3f}\t{c:.4f}\n")

    print(f"Wrote {len(cov)} rows to {out}  "
          f"(pattern cells={int(pattern.sum())}, final coverage={cov[-1]:.3f}, peak={cov.max():.3f})")


if __name__ == "__main__":
    main()
