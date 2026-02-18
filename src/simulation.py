import numpy as np
import time

from src.config import SimConfig
from src.config import NE, NW, SW, SE
from src.state import balls as bs
from src.physics import simcorexpbd as sc
from src.visualization import gltf as vis
from src.state import rods as rs

def simulation(config=None, visualization=True):
    if config is None:
        config = SimConfig()

    rodsstate = rs.RodsState(config)
    ballsstate = bs.BallsState(rodsstate, config)

    ballsstates = []
    rodsstates = []

    # Track when each ball went out of bounds (-1 means not OOB)
    oob_timestep = np.full(config.NBALL, -1, dtype=int)
    # Track when the last respawn occurred (for global cooldown)
    last_respawn_timestep = -1000  # Initialize to allow immediate first respawn

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

        if config.RESPAWN:
            oob = ((ballsstate.r[:, 0] < 0) | (ballsstate.r[:, 0] > x_max) |
                   (ballsstate.r[:, 1] < 0) | (ballsstate.r[:, 1] > y_max) |
                   (ballsstate.r[:, 2] < -0.5))

            # Mark newly out-of-bounds balls with current timestep
            newly_oob = oob & (oob_timestep == -1)
            oob_timestep[newly_oob] = timestep

            # Reset timer for balls that are back in bounds
            oob_timestep[~oob] = -1

            # Check if enough time has passed since last respawn (global cooldown)
            respawn_delay = 5.0  # seconds
            time_since_last_respawn = (timestep - last_respawn_timestep) * config.DT
            cooldown_ready = time_since_last_respawn >= respawn_delay

            # Find balls ready to respawn (OOB for at least 2 seconds)
            ready_to_respawn = oob & ((timestep - oob_timestep) * config.DT >= respawn_delay)

            if ready_to_respawn.any() and cooldown_ready:
                spawn_x, spawn_y = 0.5, 0.5
                # Check if spawn cell is empty (no in-bounds ball in the same grid cell)
                in_bounds = ~oob
                in_cell = (in_bounds
                           & (np.floor(ballsstate.r[:, 0]).astype(int) == int(np.floor(spawn_x)))
                           & (np.floor(ballsstate.r[:, 1]).astype(int) == int(np.floor(spawn_y))))
                if not in_cell.any():
                    idx = np.where(ready_to_respawn)[0][0]
                    z, _, _ = rodsstate.surfacejet(spawn_x, spawn_y)
                    ballsstate.r[idx] = [spawn_x, spawn_y, z + ballsstate.R[idx] + 0.5]
                    ballsstate.v[idx] = 0.0
                    ballsstate.w[idx] = 0.0
                    oob_timestep[idx] = -1  # Reset timer
                    last_respawn_timestep = timestep  # Update global cooldown

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
