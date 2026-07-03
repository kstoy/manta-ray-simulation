"""Quick parameter sweep for the stochastic-assembly controller.

Sweeps SA_H (how stubborn correctly-placed pattern balls are) x SA_P_OFF (how
eager off-pattern balls are to move).  Scores each run by final pattern coverage,
stray-ball count, and end-of-run settling.

    python scripts/sweep_stochastic.py
"""
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import load_experiment
from src.simulation import simulation

EXP = "experiments/mozart_stochastic.py"
STEPS = 4000          # shorter than the 8000 production run, enough to rank
COVER_RADIUS = 0.45   # a pattern cell counts as filled if a ball is this close

H_VALS = [3, 6, 10]
P_VALS = [2, 4, 7]

spec = importlib.util.spec_from_file_location("mz", EXP)
mz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mz)
BIT = mz.BIT_MAP
ny, nx = BIT.shape

base = load_experiment(EXP)
base.MAXSIMULATIONSTEPS = STEPS
D = base.D_RODS

sinks = [(cy, cx) for cy in range(ny) for cx in range(nx) if BIT[cy, cx] == '1']
sx = np.array([cx * D for (cy, cx) in sinks])
sy = np.array([cy * D for (cy, cx) in sinks])


def score(frames):
    last = frames[-1][:, :2]
    x, y = last[:, 0], last[:, 1]
    cov = sum(1 for (cy, cx) in sinks if np.min(np.hypot(x - cx * D, y - cy * D)) < COVER_RADIUS)
    strays = sum(1 for bx, by in zip(x, y) if np.min(np.hypot(bx - sx, by - sy)) > COVER_RADIUS)
    xy = frames[:, :, :2]
    move = np.linalg.norm(np.diff(xy, axis=0), axis=2).mean(axis=1)[-300:].mean()
    return cov, strays, move


results = []
print(f"sweep: {len(H_VALS)*len(P_VALS)} runs x {STEPS} steps, {len(sinks)} pattern cells\n")
for h in H_VALS:
    for p in P_VALS:
        base.SA_H = float(h)
        base.SA_P_OFF = float(p)
        base.SA_SEED = 0
        t0 = time.time()
        _, frames, _, _ = simulation(config=base, visualization=True)
        cov, strays, move = score(np.array(frames))
        dt = time.time() - t0
        results.append((h, p, cov, strays, move))
        print(f"  H={h:>2}  P_OFF={p:>2}   cov={cov:>2}/{len(sinks)}   "
              f"strays={strays:>2}   settle={move:.4f}   ({dt:.0f}s)")

print("\nranked (coverage desc, then strays asc, then settling asc):")
for h, p, cov, strays, move in sorted(results, key=lambda r: (-r[2], r[3], r[4])):
    print(f"  H={h:>2}  P_OFF={p:>2}   cov={cov:>2}/{len(sinks)}   strays={strays:>2}   settle={move:.4f}")
