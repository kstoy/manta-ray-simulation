import numpy as np
import time

from src.config import SimConfig
from src.config import NE, NW, SW, SE
from src.state import balls as bs
from src.physics import simcorexpbd as sc
from src.state import rods as rs
from src.state.balls_init import get_respawn_position, seeded_rng

def simulation(config=None, visualization=True):
    if config is None:
        config = SimConfig()

    rodsstate = rs.RodsState(config)
    ballsstate = bs.BallsState(rodsstate, config)

    ballsstates = []
    rodsstates = []
    # Per-frame per-cell scalar channels. Extend with more entries (e.g. bin_state)
    # as needed. Each entry is a list of (n_cells_x, n_cells_y) float32 arrays.
    channels = {"weight": []}

    # Track when each ball went out of bounds (-1 means not OOB)
    oob_timestep = np.full(config.NBALL, -1, dtype=int)
    # Track when the last respawn occurred (for global cooldown)
    last_respawn_timestep = -1000  # Initialize to allow immediate first respawn
    respawn_pos_fn = get_respawn_position(config.RESPAWN_STRATEGY) if config.RESPAWN_STRATEGY is not None else None
    respawn_rng = seeded_rng(config, stream=1)

    for timestep in range(config.MAXSIMULATIONSTEPS):
        rodsstate.sensors.fill(0.0)

        # Vectorized sensor aggregation
        x = ballsstate.r[:, 0]
        y = ballsstate.r[:, 1]
        m = ballsstate.m

        # Filter valid positions (within bounds)
        x_max = config.D_RODS * (config.GRIDSIZEX - 1)
        y_max = config.D_RODS * (config.GRIDSIZEY - 1)
        valid = (x > 0.0) & (x < x_max) & (y > 0.0) & (y < y_max)

        x_valid = x[valid]
        y_valid = y[valid]
        m_valid = m[valid]

        # Compute floor/ceil indices for all valid balls (in piston-grid units)
        x_idx = x_valid / config.D_RODS
        y_idx = y_valid / config.D_RODS
        x_floor = np.floor(x_idx).astype(int)
        x_ceil = np.ceil(x_idx).astype(int)
        y_floor = np.floor(y_idx).astype(int)
        y_ceil = np.ceil(y_idx).astype(int)

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
            # Slack (m) for the contact-candidate filter: balls within this distance
            # of the surface or each other are passed to the XPBD solver.
            pair_margin=0.075,
            use_grid_broadphase=True,
            linear_damping=0.01
        )

        if config.RESPAWN_STRATEGY is not None:
            # A ball is out of bounds if it has left the rod grid in (x, y) or
            # fallen below z = 0 (the floor on which the rods are mounted).
            oob = ((ballsstate.r[:, 0] < 0) | (ballsstate.r[:, 0] > x_max) |
                   (ballsstate.r[:, 1] < 0) | (ballsstate.r[:, 1] > y_max) |
                   (ballsstate.r[:, 2] < 0.0))

            # Mark newly out-of-bounds balls with current timestep
            newly_oob = oob & (oob_timestep == -1)
            oob_timestep[newly_oob] = timestep

            # Reset timer for balls that are back in bounds
            oob_timestep[~oob] = -1

            # Check if enough time has passed since last respawn (global cooldown)
            time_since_last_respawn = (timestep - last_respawn_timestep) * config.DT
            cooldown_ready = time_since_last_respawn >= config.RESPAWN_DELAY

            # Find balls ready to respawn (OOB for at least RESPAWN_DELAY seconds)
            ready_to_respawn = oob & ((timestep - oob_timestep) * config.DT >= config.RESPAWN_DELAY)

            if ready_to_respawn.any() and cooldown_ready:
                spawn_x, spawn_y = respawn_pos_fn(config, respawn_rng)
                # Optionally require the spawn cell to be empty (no in-bounds ball
                # in the same grid cell). Default for drip-through experiments where
                # balls move out of the entry cell. Set False to allow piling at a
                # fixed drop point (e.g. distributed_coverage).
                cell_clear = True
                if config.RESPAWN_REQUIRE_EMPTY_CELL:
                    in_bounds = ~oob
                    spawn_cell_x = int(np.floor(spawn_x / config.D_RODS))
                    spawn_cell_y = int(np.floor(spawn_y / config.D_RODS))
                    in_cell = (in_bounds
                               & (np.floor(ballsstate.r[:, 0] / config.D_RODS).astype(int) == spawn_cell_x)
                               & (np.floor(ballsstate.r[:, 1] / config.D_RODS).astype(int) == spawn_cell_y))
                    cell_clear = not in_cell.any()
                if cell_clear:
                    idx = np.where(ready_to_respawn)[0][0]
                    z, _, _ = rodsstate.surfacejet(spawn_x, spawn_y)
                    # Drop respawning balls from 0.25 m above the surface so they
                    # settle naturally rather than being placed in initial contact.
                    ballsstate.r[idx] = [spawn_x, spawn_y, z + ballsstate.R[idx] + 0.25]
                    ballsstate.v[idx] = 0.0
                    ballsstate.w[idx] = 0.0
                    oob_timestep[idx] = -1  # Reset timer
                    last_respawn_timestep = timestep  # Update global cooldown

        if visualization:
            rodsstates.append(rodsstate.rods.copy())
            ballsstates.append(ballsstate.r.copy())
            # Per-cell weight = NE quadrant of each piston (excluding the rightmost
            # piston column / topmost piston row, which have no NE-quadrant cell).
            n_cells_x = config.GRIDSIZEX - 1
            n_cells_y = config.GRIDSIZEY - 1
            channels["weight"].append(
                rodsstate.sensors[:n_cells_x, :n_cells_y, NE].astype(np.float32).copy()
            )

    return rodsstates, ballsstates, ballsstate.R, channels


if __name__ == "__main__":
    config = SimConfig()

    print("simulation running with visualization...", end="")
    start = time.time()
    rodsstates, ballsstates, ballsradiuses, _channels = simulation(
        config=config,
        visualization=True
    )
    end = time.time()
    print("done")
    print(f"Simulation complete - time elapsed: {end - start}")
