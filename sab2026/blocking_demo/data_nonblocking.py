"""Run the nonblocking controller experiment 10 times and save per-timestep distance metrics."""
import sys
import time
import pickle
import contextlib
import io
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimConfig
from src.controllers import CONTROLLER_REGISTRY
from src.simulation import simulation

from _common import make_clockwise_direction_map, compute_distance_metrics

DIRECTION_MAP = make_clockwise_direction_map(6, 6)

NBALL = 10
MAXSIMULATIONSTEPS = 300
N_RUNS = 100
CONTROLLER_NAME = "nonblocking"
OUTPUT_PATH = Path(__file__).parent / "data_nonblocking.pkl"


def make_config():
    dm = DIRECTION_MAP
    return SimConfig(
        GRIDSIZEX=dm.shape[1] + 1,
        GRIDSIZEY=dm.shape[0] + 1,
        MAXSIMULATIONSTEPS=MAXSIMULATIONSTEPS,
        BALL_INIT="random",
        RESPAWN_STRATEGY="random",
        RESPAWN_DELAY=0.5,
        CONTROLLER=lambda cfg, d=dm: CONTROLLER_REGISTRY[CONTROLLER_NAME](cfg, d),
        NBALL=NBALL,
    )


def run_once(config):
    with contextlib.redirect_stdout(io.StringIO()):
        _, ballsstates, _, _ = simulation(config=config, visualization=True)
    return compute_distance_metrics(ballsstates, config)


if __name__ == "__main__":
    pair_runs = np.zeros((N_RUNS, MAXSIMULATIONSTEPS))
    center_runs = np.zeros((N_RUNS, MAXSIMULATIONSTEPS))

    print(f"Running {N_RUNS} {CONTROLLER_NAME} simulations...")
    total_start = time.time()
    for i in range(N_RUNS):
        config = make_config()
        start = time.time()
        pair_dist, center_dist = run_once(config)
        pair_runs[i] = pair_dist
        center_runs[i] = center_dist
        print(f"  run {i+1}/{N_RUNS} done ({time.time() - start:.1f}s)")
    print(f"Total time: {time.time() - total_start:.1f}s")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "controller": CONTROLLER_NAME,
            "n_runs": N_RUNS,
            "nball": NBALL,
            "timesteps": MAXSIMULATIONSTEPS,
            "pair_dist": pair_runs,      # shape (N_RUNS, T)
            "center_dist": center_runs,  # shape (N_RUNS, T)
        }, f)
    print(f"Saved to {OUTPUT_PATH}")
