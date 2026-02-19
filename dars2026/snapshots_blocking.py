"""Run 9x9 center-pointing simulation with blocking controller and save 8 snapshot images."""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimConfig
from src.controllers import CONTROLLER_REGISTRY
from src.simulation import simulation
from src.visualization.matplotlib import _compute_surface_grid

DIRECTION_MAP = np.flip(np.array([
    ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'S'],
    ['N', 'E', 'E', 'E', 'E', 'E', 'S', 'S'],
    ['N', 'N', 'E', 'E', 'E', 'S', 'S', 'S'],
    ['N', 'N', 'N', 'E', 'S', 'S', 'S', 'S'],
    ['N', 'N', 'N', 'N', 'W', 'S', 'S', 'S'],
    ['N', 'N', 'N', 'W', 'W', 'W', 'S', 'S'],
    ['N', 'N', 'W', 'W', 'W', 'W', 'W', 'S'],
    ['N', 'W', 'W', 'W', 'W', 'W', 'W', 'S']
]), 0)

NBALL = 20
MAXSIMULATIONSTEPS = 300
N_SNAPSHOTS = 8
OUTPUT_DIR = Path(__file__).parent / "snapshots_blocking"


def render_frame(rods, balls, ballradiuses, config, step, path, resolution=20):
    """Render a single simulation frame and save as PNG."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = _compute_surface_grid(rods, config, resolution)
    ax.plot_surface(X, Y, Z, color='green', alpha=0.6, edgecolor='none')

    sizes = (ballradiuses * 80) ** 2
    ax.scatter(balls[:, 0], balls[:, 1], balls[:, 2],
               c='red', s=sizes, depthshade=True)

    x_max = (config.GRIDSIZEX - 1) * config.D
    y_max = (config.GRIDSIZEY - 1) * config.D
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_zlim(-0.5, 2.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title(f'Blocking — Step {step}')

    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    dm = DIRECTION_MAP
    config = SimConfig(
        GRIDSIZEX=dm.shape[1] + 1,
        GRIDSIZEY=dm.shape[0] + 1,
        MAXSIMULATIONSTEPS=MAXSIMULATIONSTEPS,
        BALL_INIT="random",
        RESPAWN=True,
        CONTROLLER=lambda cfg, d=dm: CONTROLLER_REGISTRY["blocking"](cfg, d),
        NBALL = 20
    )

    print("Running blocking simulation...", end="", flush=True)
    start = time.time()
    rodsstates, ballsstates, ballsradiuses = simulation(config=config, visualization=True)
    print(f" done ({time.time() - start:.1f}s)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(rodsstates)
    indices = np.linspace(0, total - 1, N_SNAPSHOTS, dtype=int)

    for k, idx in enumerate(indices):
        out_path = OUTPUT_DIR / f"frame_{k:02d}_step{idx:04d}.png"
        print(f"  Rendering frame {k+1}/{N_SNAPSHOTS} (step {idx})...", end="", flush=True)
        render_frame(rodsstates[idx], ballsstates[idx], ballsradiuses, config, idx, out_path)
        print(" saved")

    print(f"Snapshots saved to {OUTPUT_DIR}")
