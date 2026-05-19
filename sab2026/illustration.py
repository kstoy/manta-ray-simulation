"""Render a single illustration of a 2x2 cell system (3x3 = 9 pistons, 1 sphere).

Style matches sab2026/snapshots_blocking — same OpenGL renderer, same colors.
"""
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SimConfig
from src.physics.catenarysurface import jet1
from src.visualization.opengl_video import OpenGLVideoExporter

HERE = Path(__file__).parent
OUTPUT_PATH = HERE / "illustration.png"


if __name__ == "__main__":
    config = SimConfig(
        GRIDSIZEX=3,
        GRIDSIZEY=3,
        D_RODS=0.5,
        D_FABRIC=0.6,
        LOW_HEIGHT=0.7,
        HIGH_HEIGHT=1.0,
        BALL_RADIUS=0.05,
        NBALL=1,
    )

    rods = np.zeros((config.GRIDSIZEX, config.GRIDSIZEY, 3), dtype=np.float32)
    for i in range(config.GRIDSIZEX):
        for j in range(config.GRIDSIZEY):
            rods[i, j, 0] = i * config.D_RODS
            rods[i, j, 1] = j * config.D_RODS
            rods[i, j, 2] = config.HIGH_HEIGHT

    grid_center_x = (config.GRIDSIZEX - 1) * config.D_RODS / 2
    grid_center_y = (config.GRIDSIZEY - 1) * config.D_RODS / 2
    ball_x = grid_center_x + config.D_RODS / 2
    ball_y = grid_center_y - config.D_RODS / 2

    ci = min(int(ball_x / config.D_RODS), config.GRIDSIZEX - 2)
    cj = min(int(ball_y / config.D_RODS), config.GRIDSIZEY - 2)
    cj = max(cj, 0)
    rodheights = (rods[ci, cj, 2], rods[ci + 1, cj, 2],
                  rods[ci, cj + 1, 2], rods[ci + 1, cj + 1, 2])
    surface_z, _, _ = jet1(ball_x - ci * config.D_RODS,
                           ball_y - cj * config.D_RODS,
                           rodheights, config.D_RODS, config.D_FABRIC)
    ball_pos = np.array([[ball_x, ball_y, surface_z + config.BALL_RADIUS]],
                        dtype=np.float32)

    exporter = OpenGLVideoExporter(
        rodsstates=[rods],
        ballsstates=[ball_pos],
        ballradiuses=np.array([config.BALL_RADIUS], dtype=np.float32),
        config=config,
        resolution=20,
        width=1200,
        height=800,
        fps=30,
        rod_radius=0.012,
    )
    exporter.camera_distance = 1.5
    exporter.camera_angle_v = np.radians(45)
    exporter.camera_target = np.array(
        [grid_center_x, grid_center_y, 0.45], dtype=np.float32
    )
    exporter.init_gl()
    image = exporter.render_frame(0)

    imageio.imwrite(OUTPUT_PATH, image)
    print(f"Saved illustration to {OUTPUT_PATH}")
