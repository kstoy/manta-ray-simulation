"""Re-run one sab_demo seed with visualization data and export an MP4.

Reuses build_config() from sab_demo_convergence so the run matches the
convergence study exactly (perimeter_random init/respawn + given SEED).

Usage:
    python sab2026/sab_demo/render_seed.py 5            # render seed 5 -> output/sab_demo_seed5.mp4
    python sab2026/sab_demo/render_seed.py 5 --output foo.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sab_demo_convergence import build_config  # noqa: E402
from src.simulation import simulation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, help="seed to render")
    parser.add_argument("--output", default=None, help="MP4 path")
    args = parser.parse_args()

    output = args.output or f"output/sab_demo_seed{args.seed}.mp4"
    Path("output").mkdir(exist_ok=True)

    config = build_config(args.seed)
    print(f"Simulating seed {args.seed} ({config.MAXSIMULATIONSTEPS} steps)...", flush=True)
    rodsstates, ballsstates, ballsradiuses, channels = simulation(config=config, visualization=True)

    from src.visualization.opengl_video import export_video
    print("Exporting video...", flush=True)
    export_video(rodsstates, ballsstates, ballsradiuses, config,
                 output_path=output, fps=30, channels=channels)
    print(f"Video written to {output}")


if __name__ == "__main__":
    main()
