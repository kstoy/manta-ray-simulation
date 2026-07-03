"""Run 5x5 circular simulation with nonblocking controller and save 8 snapshot images."""
import sys
import time
import numpy as np
import imageio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimConfig
from src.controllers import CONTROLLER_REGISTRY
from src.simulation import simulation
from src.visualization.opengl_video import OpenGLVideoExporter
import glfw

from _common import make_clockwise_direction_map

DIRECTION_MAP = make_clockwise_direction_map(6, 6)

NBALL = 10
MAXSIMULATIONSTEPS = 300
N_SNAPSHOTS = 8
OUTPUT_DIR = Path(__file__).parent / "snapshots_nonblocking"
VIDEO_PATH = OUTPUT_DIR / "nonblocking.mp4"


if __name__ == "__main__":
    dm = DIRECTION_MAP
    config = SimConfig(
        GRIDSIZEX=dm.shape[1] + 1,
        GRIDSIZEY=dm.shape[0] + 1,
        MAXSIMULATIONSTEPS=MAXSIMULATIONSTEPS,
        BALL_INIT="random",
        RESPAWN_STRATEGY="random",
        RESPAWN_DELAY=0.5,
        CONTROLLER=lambda cfg, d=dm: CONTROLLER_REGISTRY["nonblocking"](cfg, d),
        NBALL=NBALL,
    )

    print("Running nonblocking simulation...", end="", flush=True)
    start = time.time()
    rodsstates, ballsstates, ballsradiuses, _channels = simulation(config=config, visualization=True)
    print(f" done ({time.time() - start:.1f}s)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(rodsstates)
    indices = np.linspace(0, total - 1, N_SNAPSHOTS, dtype=int)

    renderer = OpenGLVideoExporter(rodsstates, ballsstates, ballsradiuses, config, resolution=10, width=1200, height=800)
    renderer.camera_distance = max(config.GRIDSIZEX, config.GRIDSIZEY) * config.D_RODS * 1.0
    renderer.camera_angle_v = np.radians(35)
    renderer.init_gl()

    for k, idx in enumerate(indices):
        t = idx * config.DT
        out_path = OUTPUT_DIR / f"frame_{k:02d}_t{t:05.1f}s.png"
        print(f"  Rendering frame {k+1}/{N_SNAPSHOTS} (t={t:.1f}s)...", end="", flush=True)
        image = renderer.render_frame(idx)
        imageio.imwrite(str(out_path), image)
        print(" saved")

    print(f"Exporting video...", end="", flush=True)
    writer = imageio.get_writer(str(VIDEO_PATH), fps=30, codec='libx264', pixelformat='yuv420p', quality=8)
    for frame in range(total):
        writer.append_data(renderer.render_frame(frame))
    writer.close()
    print(f" saved to {VIDEO_PATH}")

    glfw.terminate()
    print(f"Snapshots saved to {OUTPUT_DIR}")
