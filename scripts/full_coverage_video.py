"""Find a run that reaches 100% pattern coverage, then render a 15s clip of it.

Re-runs the experiment across seeds until one run covers all pattern cells, then
clips that run at the first step it hits 100% and subsamples to exactly 15s.

    python scripts/full_coverage_video.py
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import load_experiment
from src.simulation import simulation
from src.visualization.opengl_video import export_video

EXP = "experiments/mozart_stochastic.py"
STEPS = 5000
DURATION = 15.0
FPS = 30
N_FRAMES = int(DURATION * FPS)        # 450
MAX_ATTEMPTS = 6
OUT = "output/mozart_100.mp4"

spec = importlib.util.spec_from_file_location("mz", EXP)
mz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mz)
pattern = (np.asarray(mz.BIT_MAP).astype(str) == '1')
ny, nx = pattern.shape

cfg = load_experiment(EXP)
cfg.MAXSIMULATIONSTEPS = STEPS
D = cfg.D_RODS
n_cells = int(pattern.sum())


def cov_traj(balls):
    """Per-frame count of pattern cells containing >=1 ball (ball binned to its cell)."""
    out = np.empty(len(balls), int)
    for t, f in enumerate(balls):
        f = np.asarray(f)
        cj = np.floor(f[:, 0] / D).astype(int)
        ci = np.floor(f[:, 1] / D).astype(int)
        valid = (ci >= 0) & (ci < ny) & (cj >= 0) & (cj < nx)
        occ = np.zeros((ny, nx), dtype=bool)
        occ[ci[valid], cj[valid]] = True
        out[t] = int(occ[pattern].sum())
    return out


winner = None          # first run to hit 100%
best = None            # highest-peak run seen, as a fallback
for attempt in range(MAX_ATTEMPTS):
    cfg.SA_SEED = attempt
    rods, balls, radii, _ = simulation(config=cfg, visualization=True)
    cov = cov_traj(balls)
    peak = int(cov.max())
    if best is None or peak > best[0]:
        best = (peak, rods, balls, radii, int(cov.argmax()))
    if peak >= n_cells:
        hit = int(np.argmax(cov >= n_cells))
        print(f"attempt {attempt}: HIT 100% ({n_cells}/{n_cells}) at step {hit}", flush=True)
        winner = (rods, balls, radii, hit)
        break
    print(f"attempt {attempt}: peak {peak}/{n_cells} at step {int(cov.argmax())}", flush=True)

if winner is not None:
    rods, balls, radii, hit = winner
else:
    peak, rods, balls, radii, hit = best
    print(f"No 100% run in {MAX_ATTEMPTS} attempts; rendering best = "
          f"{peak}/{n_cells} (clipped at its peak step {hit}).", flush=True)
idxs = np.linspace(0, hit, num=N_FRAMES).round().astype(int)
rods_s = [rods[i] for i in idxs]
balls_s = [balls[i] for i in idxs]
print(f"rendering {len(rods_s)} frames (steps 0..{hit}) -> {OUT} "
      f"= {len(rods_s) / FPS:.1f}s @ {FPS}fps", flush=True)
export_video(rods_s, balls_s, radii, cfg, output_path=OUT, fps=FPS,
             resolution=7, channels=None)
print("done", flush=True)
