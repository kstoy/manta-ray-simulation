#!/usr/bin/env python
"""Run simulation and display results with matplotlib 3D visualization."""
import time

from src.simulation import simulation
from src.config import SimConfig
from src.visualization_matplotlib import animate_simulation

if __name__ == "__main__":
    config = SimConfig()

    print("running simulation...", end="")
    start = time.time()
    rodsstates, ballsstates, ballsradiuses = simulation(
        config=config,
        visualization=True
    )
    end = time.time()
    print("done")
    print(f"Simulation complete - time elapsed: {end - start}")

    print("launching matplotlib visualization...")
    animate_simulation(rodsstates, ballsstates, ballsradiuses, config)
