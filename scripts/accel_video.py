"""Render an accelerated video of the MOZART stochastic assembly up to a step.

Runs the current experiment to STOP steps, subsamples frames so the clip plays
in ~DURATION seconds, and exports an MP4.

    python scripts/accel_video.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import load_experiment
from src.simulation import simulation
from src.visualization.opengl_video import export_video

EXP = "experiments/mozart_stochastic.py"
STOP = 6300          # last simulation step to show
DURATION = 10.0      # target clip length in seconds
FPS = 30
OUT = "output/mozart_accel.mp4"

n_frames = int(DURATION * FPS)               # 300
step = max(1, STOP // n_frames)              # 21 -> every 21st step

cfg = load_experiment(EXP)
cfg.MAXSIMULATIONSTEPS = STOP
print(f"running {STOP} steps of {EXP} ...")
rods, balls, radii, _ = simulation(config=cfg, visualization=True)

rods_s = rods[::step]
balls_s = balls[::step]
print(f"rendering {len(rods_s)} frames -> {OUT} "
      f"({len(rods_s) / FPS:.1f}s @ {FPS}fps, every {step}th step)")
export_video(rods_s, balls_s, radii, cfg, output_path=OUT, fps=FPS,
             resolution=7, channels=None)
print("done")
