import numpy as np
import pickle
import time
import sys
import contextlib
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimConfig
from src.controllers import CONTROLLER_REGISTRY
from src import simulation as sim

SURFACE_SIZES = [(5, 5), (9, 9), (17, 17)]
BALL_COUNTS = list(range(10, 110, 10))
REPETITIONS = 10
TIMESTEPS = 300
RESULTS_PATH = Path(__file__).parent / "benchmark_results.pkl"

def make_clockwise_direction_map(n_rows, n_cols):
    dmap = np.empty((n_rows, n_cols), dtype='<U1')
    cy = (n_rows - 1) / 2.0
    cx = (n_cols - 1) / 2.0
    for j in range(n_rows):
        for i in range(n_cols):
            dy = j - cy
            dx = i - cx
            if dy > 0:   # tie, upper half: NW->E, NE->S
                dmap[j, i] = 'E' if dx < 0 else 'S'
            else:          # tie, lower half: SW->N, SE->W
                dmap[j, i] = 'N' if dx < 0 else 'W'
    return dmap


def make_config(gx, gy, nball):
    """Create a SimConfig with nonblocking clockwise controller."""
    dm = make_clockwise_direction_map(gy - 1, gx - 1)
    config = SimConfig(
        GRIDSIZEX=gx,
        GRIDSIZEY=gy,
        MAXSIMULATIONSTEPS=TIMESTEPS,
        BALL_INIT="random",
        RESPAWN_STRATEGY="random",
        CONTROLLER=lambda cfg, d=dm: CONTROLLER_REGISTRY["nonblocking"](cfg, d),
    )
    config.NBALL = nball
    return config


def run_simulation_quiet(config):
    """Run simulation with stdout suppressed (avoids controller print spam)."""
    with contextlib.redirect_stdout(io.StringIO()):
        sim.simulation(config=config, visualization=False)


def run_benchmarks():
    # Warm-up run to avoid cold-start bias
    print("Warm-up run...", flush=True)
    warmup_config = make_config(*SURFACE_SIZES[0], nball=10)
    run_simulation_quiet(warmup_config)

    results = {}

    for gx, gy in SURFACE_SIZES:
        label = f"{gx}x{gy}"
        means = []
        stds = []

        for nball in BALL_COUNTS:
            config = make_config(gx, gy, nball)

            times = []
            for _ in range(REPETITIONS):
                start = time.time()
                run_simulation_quiet(config)
                elapsed = time.time() - start
                times.append(elapsed)

            avg = np.mean(times)
            std = np.std(times)
            means.append(avg)
            stds.append(std)
            print(f"{label}  balls={nball:3d}  mean={avg:.3f}s  std={std:.3f}s")

        results[label] = (np.array(means), np.array(stds))
        print()

    return results


if __name__ == "__main__":
    dm = make_clockwise_direction_map(4, 4)
    print("Direction map for 5x5 grid:")
    print("  " + " ".join(str(i) for i in range(dm.shape[1])))
    for j in range(dm.shape[0] - 1, -1, -1):
        print(f"{j} " + " ".join(dm[j]))
    print()

    results = run_benchmarks()
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({
            "results": results,
            "ball_counts": BALL_COUNTS,
            "surface_sizes": SURFACE_SIZES,
            "repetitions": REPETITIONS,
            "timesteps": TIMESTEPS,
        }, f)
    print(f"Results saved to {RESULTS_PATH}")
