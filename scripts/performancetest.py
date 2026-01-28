import numpy as np
import time
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimConfig
from src import simulation as sim


def runexperiment(config):
    for ballcount in range(0, 105, 5):
        test_config = replace(config, GRIDSIZEX=ballcount + 1)

        times = []
        for _ in range(10):
            start = time.time()
            sim.simulation(
                config=test_config,
                visualization=False
            )
            end = time.time()
            times.append(end - start)

        avg = np.average(times)
        std = np.std(times)
        print(f"{ballcount} {avg} {std}")

    print("\n\n")


if __name__ == "__main__":
    print("# plot \"timevsballs.dat\" index 0 title \"5x5 surface\" with errorlines, "
          "\"timevsballs.dat\" index 1 title \"10x10 surface\" with errorlines")

    print("# 5x5, 100 timesteps, uniform")
    runexperiment(SimConfig(GRIDSIZEX=5, GRIDSIZEY=5))

    print("# 10x10, 100 timesteps, uniform")
    runexperiment(SimConfig(GRIDSIZEX=10, GRIDSIZEY=10))
