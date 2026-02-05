#!/usr/bin/env python
"""Run simulation with a custom direction map."""
import time
import numpy as np

from src.simulation import simulation
from src.config import SimConfig
from src import visualization as vis
from src.controllers.direction_map_controller_adaptive import DirectionMapControllerAdaptive

# =============================================================================
# EDIT YOUR DIRECTION MAP HERE
# =============================================================================
# Map size should be (GRIDSIZEY-1, GRIDSIZEX-1)
# For a 5x5 grid of pistons, you need a 4x4 map
#
# Directions: N, S, E, W, I (idle)
# Priority: "NE" means prefer N, fallback to E if blocked
# Row 0 is BOTTOM, last row is TOP

DIRECTION_MAP = np.flip( np.array([
    ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I',  'I', 'I'],  # row 7
    ['I', 'I', 'I', 'I', 'W', 'E', 'I', 'I', 'I', 'I', 'I',  'I', 'I'],  # row 7
    ['N', 'N', 'I', 'I', 'WN','NE','I', 'I', 'N', 'N', 'I',  'N', 'N'],  # row 6
    ['N', 'N', 'I', 'I', 'N', 'N', 'I', 'I', 'N', 'N', 'I',  'N', 'N'],  # row 4 (middle)
    ['N', 'N', 'I', 'I', 'N', 'N', 'I', 'I', 'N', 'N', 'I',  'N', 'N'],  # row 2
    ['N', 'N', 'I', 'I', 'N', 'N', 'I', 'I', 'N', 'N', 'N',  'N', 'N'],  # row 1
    ['NE','NE','E', 'E', 'NE','NE','E', 'E', 'NE','NE','NE', 'NE','N'],  # row 0 (bottom)
]), 0 )

# =============================================================================
# SIMULATION CONFIG
# =============================================================================
GRIDSIZEX = 14  # Must be DIRECTION_MAP columns + 1
GRIDSIZEY = 8  # Must be DIRECTION_MAP rows + 1


class CustomDirectionMapController(DirectionMapControllerAdaptive):
    """Controller with pre-defined custom direction map."""

    def __init__(self, config, direction_map=None):
        # Use our custom map instead of default
        super().__init__(config, direction_map=DIRECTION_MAP)


if __name__ == "__main__":
    # Validate map size
    expected_rows = GRIDSIZEY - 1
    expected_cols = GRIDSIZEX - 1
    actual_rows, actual_cols = DIRECTION_MAP.shape

    if actual_rows != expected_rows or actual_cols != expected_cols:
        print(f"ERROR: Direction map size mismatch!")
        print(f"  Expected: ({expected_rows}, {expected_cols})")
        print(f"  Got: ({actual_rows}, {actual_cols})")
        print(f"  Grid size is {GRIDSIZEX}x{GRIDSIZEY}, so map should be {expected_cols}x{expected_rows}")
        exit(1)

    # Register our custom controller temporarily
    from src.controllers import CONTROLLER_REGISTRY
    CONTROLLER_REGISTRY["custom_direction_map"] = CustomDirectionMapController

    config = SimConfig(
        GRIDSIZEX=GRIDSIZEX,
        GRIDSIZEY=GRIDSIZEY,
        CONTROLLER="custom_direction_map",
    )
    config.NBALL = 46  # Set number of balls

    print("Running simulation with custom direction map...")
    start = time.time()
    rodsstates, ballsstates, ballsradiuses = simulation(
        config=config,
        visualization=True
    )
    end = time.time()
    print(f"Simulation complete - time elapsed: {end - start:.2f}s")

    vis.generategltffiles("surfacevisualization", rodsstates, ballsstates, ballsradiuses, config)
    print("Output written to output/surfacevisualization.gltf")
