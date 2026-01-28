import numpy as np
import time

from src.config import SimConfig
from src.constants import NE, NW, SW, SE
from src import ballstate as bs
from src.physics import simcorexpbd as sc
from src import visualization as vis
from src import rodstate as rs

def simulation(config=None, visualization=True):
    if config is None:
        config = SimConfig()

    rodsstate = rs.RodsState(config)
    ballsstate = bs.BallsState(rodsstate, config)

    ballsstates = []
    rodsstates = []

    for timestep in range(config.MAXSIMULATIONSTEPS):
        rodsstate.sensors.fill(0.0)

        # Vectorized sensor aggregation
        x = ballsstate.r[:, 0]
        y = ballsstate.r[:, 1]
        m = ballsstate.m

        # Filter valid positions (within bounds)
        x_max = config.D * (config.GRIDSIZEX - 1)
        y_max = config.D * (config.GRIDSIZEY - 1)
        valid = (x > 0.0) & (x < x_max) & (y > 0.0) & (y < y_max)

        x_valid = x[valid]
        y_valid = y[valid]
        m_valid = m[valid]

        # Compute floor/ceil indices for all valid balls
        x_floor = np.floor(x_valid).astype(int)
        x_ceil = np.ceil(x_valid).astype(int)
        y_floor = np.floor(y_valid).astype(int)
        y_ceil = np.ceil(y_valid).astype(int)

        # Accumulate masses at the 4 corners (NE, NW, SW, SE)
        np.add.at(rodsstate.sensors, (x_floor, y_floor, NE), m_valid)
        np.add.at(rodsstate.sensors, (x_ceil, y_floor, NW), m_valid)
        np.add.at(rodsstate.sensors, (x_ceil, y_ceil, SW), m_valid)
        np.add.at(rodsstate.sensors, (x_floor, y_ceil, SE), m_valid)

        rodsstate.settimestep(timestep)
        rodsstate.update()

        sc.step(
            ballsstate,
            rodsstate,
            dt=config.DT,
            gravity=9.81,
            mu_s=0.5,
            mu_k=0.5,
            compliance_n=1e-8,
            num_pos_iters=5,
            substeps=1,
            pair_margin=0.15,
            use_grid_broadphase=True,
            linear_damping=0.01
        )
        if visualization:
            rodsstates.append(rodsstate.rods.copy())
            ballsstates.append(ballsstate.r.copy())

    return rodsstates, ballsstates, ballsstate.R


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
