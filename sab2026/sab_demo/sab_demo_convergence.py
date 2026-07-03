"""Convergence statistics for sab_demo.py over many randomized runs.

For each run we use the `perimeter_random` init + respawn (seeded by run index)
so runs differ, then:
  - count objects per cell per timestep (ball center inside the cell),
  - decide success/fail: a run SUCCEEDS if, at the final timestep, every target
    cell (see target_map.py) holds >= 1 ball,
  - for successes, report the STABLE convergence timestep: the earliest t0 after
    which every target cell stays occupied continuously until the end.

Runs are embarrassingly parallel and executed across a process pool.

Usage:
    python sab2026/sab_demo/sab_demo_convergence.py                 # 100 runs, all cores
    python sab2026/sab_demo/sab_demo_convergence.py --runs 20       # fewer runs
    python sab2026/sab_demo/sab_demo_convergence.py --workers 4     # cap parallelism
"""
import argparse
import contextlib
import io
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run import load_experiment  # noqa: E402
from src.simulation import simulation  # noqa: E402
from target_map import target_mask  # noqa: E402

SAB_DEMO = Path(__file__).parent / "sab_demo.py"
OUT_PKL = Path(__file__).parent / "sab_demo_convergence.pkl"
DT = 0.1  # seconds per timestep (matches SimConfig.DT); filled from config at runtime


def build_config(seed):
    """Base config from sab_demo.py, overridden with randomized perimeter
    init/respawn and a per-run seed."""
    config = load_experiment(str(SAB_DEMO))
    config.BALL_INIT = "perimeter_random"
    config.RESPAWN_STRATEGY = "perimeter_random"
    config.SEED = int(seed)
    return config


def compute_counts(ballsstates, config):
    """Objects per cell per timestep -> int16 array (T, n_cells_x, n_cells_y).

    A ball belongs to cell (cx, cy) when floor(x/D_RODS)==cx and
    floor(y/D_RODS)==cy (center-based), matching the sensor convention.
    """
    ncx = config.GRIDSIZEX - 1
    ncy = config.GRIDSIZEY - 1
    T = len(ballsstates)
    counts = np.zeros((T, ncx, ncy), dtype=np.int16)
    inv = 1.0 / config.D_RODS
    for t, r in enumerate(ballsstates):
        cx = np.floor(r[:, 0] * inv).astype(int)
        cy = np.floor(r[:, 1] * inv).astype(int)
        valid = (cx >= 0) & (cx < ncx) & (cy >= 0) & (cy < ncy)
        np.add.at(counts[t], (cx[valid], cy[valid]), 1)
    return counts


def convergence(counts, txs, tys):
    """Return (success, t0, occupied_targets_per_t).

    occupied_targets_per_t[t] = number of target cells with >= 1 ball at t.
    success = all targets occupied at the final timestep.
    t0 = earliest timestep after which all targets stay occupied to the end
         (stable convergence); -1 if the run never stably converges.
    """
    n_targets = len(txs)
    occupied = (counts[:, txs, tys] >= 1).sum(axis=1)  # (T,)
    converged = occupied == n_targets
    if not converged[-1]:
        return False, -1, occupied
    not_conv = np.where(~converged)[0]
    t0 = 0 if not_conv.size == 0 else int(not_conv[-1] + 1)
    return True, t0, occupied


def run_one(seed):
    """Execute one simulation and return its convergence result + per-cell counts."""
    mask = target_mask()             # (n_cell_rows, n_cell_cols) = [cy, cx]
    tys, txs = np.where(mask)         # target cell rows (cy) and cols (cx)
    config = build_config(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        _rods, ballsstates, _radii, _ch = simulation(config=config, visualization=True)
    counts = compute_counts(ballsstates, config)
    success, t0, occupied = convergence(counts, txs, tys)
    return {
        "seed": int(seed),
        "success": bool(success),
        "t0": int(t0),
        "t0_seconds": (t0 * config.DT) if t0 >= 0 else float("nan"),
        "occupied_per_t": occupied.astype(np.int16),
        "counts": counts,  # (T, ncx, ncy) int16
        "dt": config.DT,
    }


def summarize(results, n_targets):
    successes = [r for r in results if r["success"]]
    n = len(results)
    ns = len(successes)
    print(f"\n=== Convergence over {n} runs ({n_targets} target cells) ===")
    print(f"Success rate: {ns}/{n} = {100.0 * ns / n:.1f}%")
    if successes:
        t0 = np.array([r["t0"] for r in successes])
        dt = results[0]["dt"]
        sec = t0 * dt
        print("\nStable convergence timestep (successful runs):")
        print(f"  mean   = {t0.mean():.1f} steps  ({sec.mean():.2f} s)")
        print(f"  std    = {t0.std(ddof=1):.1f} steps  ({sec.std(ddof=1):.2f} s)" if ns > 1 else "  std    = n/a")
        print(f"  median = {np.median(t0):.1f} steps  ({np.median(sec):.2f} s)")
        print(f"  min    = {t0.min()} steps  ({sec.min():.2f} s)")
        print(f"  max    = {t0.max()} steps  ({sec.max():.2f} s)")
    failures = [r["seed"] for r in results if not r["success"]]
    if failures:
        print(f"\nFailed seeds ({len(failures)}): {failures}")


def make_plots(results, mask):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping plots: matplotlib unavailable: {e})")
        return

    outdir = Path(__file__).parent
    n_targets = int(mask.sum())
    successes = [r for r in results if r["success"]]

    # 1. Convergence-time histogram (successful runs).
    if successes:
        t0 = np.array([r["t0"] for r in successes]) * results[0]["dt"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(t0, bins=min(20, max(5, len(t0) // 3)), color="#3477eb", edgecolor="k")
        ax.set_xlabel("Stable convergence time (s)")
        ax.set_ylabel("Number of runs")
        ax.set_title(f"Convergence time ({len(successes)} successful runs)")
        fig.tight_layout()
        fig.savefig(outdir / "convergence_time_hist.pdf")
        plt.close(fig)

    # 2. Convergence curve: fraction of target cells occupied vs timestep.
    occ = np.stack([r["occupied_per_t"] for r in results]).astype(float) / n_targets
    mean = occ.mean(axis=0)
    std = occ.std(axis=0)
    t = np.arange(occ.shape[1]) * results[0]["dt"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, mean, color="#eb4034", label="mean")
    ax.fill_between(t, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                    color="#eb4034", alpha=0.2, label="std. dev.")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction of target cells occupied")
    ax.set_title(f"Target occupancy over time ({len(results)} runs)")
    ax.set_xlim(0, 450)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "convergence_curve.pdf")
    plt.close(fig)

    print(f"\nPlots saved to {outdir}/ (convergence_time_hist.pdf, "
          f"convergence_curve.pdf)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100, help="number of runs")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel processes (default: all cores)")
    parser.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    args = parser.parse_args()

    mask = target_mask()
    n_targets = int(mask.sum())
    workers = args.workers or os.cpu_count() or 1
    seeds = list(range(args.runs))

    print(f"Running {args.runs} sab_demo runs on {workers} workers "
          f"({n_targets} target cells)...")
    start = time.time()
    results = [None] * args.runs
    with Pool(processes=workers) as pool:
        for done, res in enumerate(pool.imap_unordered(run_one, seeds), 1):
            results[res["seed"]] = res
            flag = "OK " if res["success"] else "FAIL"
            t0s = f"{res['t0_seconds']:.1f}s" if res["success"] else "-"
            print(f"  [{done:3d}/{args.runs}] seed={res['seed']:3d} {flag} t0={t0s}",
                  flush=True)
    print(f"\nAll runs done in {time.time() - start:.1f}s")

    summarize(results, n_targets)

    with open(OUT_PKL, "wb") as f:
        pickle.dump({
            "results": results,
            "target_mask": mask,
            "n_targets": n_targets,
            "runs": args.runs,
        }, f)
    print(f"\nResults saved to {OUT_PKL}")

    if not args.no_plots:
        make_plots(results, mask)


if __name__ == "__main__":
    main()
