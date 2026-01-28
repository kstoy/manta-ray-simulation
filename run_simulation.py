#!/usr/bin/env python
"""Main entry point for running the surface simulation."""
import time

from src.simulation import simulation
from src.config import SimConfig
from src import visualization as vis

if __name__ == "__main__":
    config = SimConfig()

    print("simulation running with visualization...", end="")
    start = time.time()
    rodsstates, ballsstates, ballsradiuses = simulation(
        config=config,
        visualization=True
    )
    end = time.time()
    print("done")
    print(f"Simulation complete - time elapsed: {end - start}")

    vis.generategltffiles("surfacevisualization", rodsstates, ballsstates, ballsradiuses, config)
