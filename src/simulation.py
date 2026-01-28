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

        for r, m in zip(ballsstate.r, ballsstate.m):
            x, y = r[0], r[1]
            if 0.0 < x < config.D * (config.GRIDSIZEX - 1) and 0.0 < y < config.D * (config.GRIDSIZEY - 1):
                rodsstate.sensors[int(np.floor(x))][int(np.floor(y))][NE] += m
                rodsstate.sensors[int(np.ceil(x))][int(np.floor(y))][NW] += m
                rodsstate.sensors[int(np.ceil(x))][int(np.ceil(y))][SW] += m
                rodsstate.sensors[int(np.floor(x))][int(np.ceil(y))][SE] += m

        rodsstate.settimestep(timestep)
        rodsstate.update()

        if visualization:
            rodsstates.append(rodsstate.rods.copy())

        sc.step(
            ballsstate,
            rodsstate,
            dt=config.DT,
            gravity=9.81,
            mu_s=0.5,
            mu_k=0.5,
            compliance_n=1e-8,
            num_pos_iters=10,
            substeps=2,
            pair_margin=0.15,
            use_grid_broadphase=False,
            linear_damping=0.01
        )

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
