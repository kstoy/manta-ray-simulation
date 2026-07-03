#!/usr/bin/env python
"""Visualize simulation data from a saved data file.

Usage:
    python visualize.py opengl           # interactive 3D viewer
    python visualize.py video            # export output/simulation.mp4
    python visualize.py opengl --input mydata.pkl  # custom data file path
"""
import argparse
from pathlib import Path

from src import simdata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize manta ray simulation data")
    parser.add_argument(
        "viz", choices=["opengl", "video"],
        help="Visualization mode"
    )
    parser.add_argument(
        "--input", default=simdata.DEFAULT_PATH,
        help=f"Data file path (default: {simdata.DEFAULT_PATH})"
    )
    args = parser.parse_args()

    data = simdata.load(args.input)
    rodsstates = data["rodsstates"]
    ballsstates = data["ballsstates"]
    ballsradiuses = data["ballsradiuses"]
    config = data["config"]

    if args.viz == "opengl":
        from src.visualization.opengl import animate_simulation
        print("Launching OpenGL visualization...")
        animate_simulation(rodsstates, ballsstates, ballsradiuses, config)

    elif args.viz == "video":
        from src.visualization.opengl_video import export_video
        output_path = "output/simulation.mp4"
        Path("output").mkdir(exist_ok=True)
        print("Exporting video...")
        export_video(rodsstates, ballsstates, ballsradiuses, config,
                     output_path=output_path, fps=30)
        print(f"Video written to {output_path}")
