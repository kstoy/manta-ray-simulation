"""Phased greedy bin-filling demo.

Same 4x11 topology as greedy_bin_demo: balls drop in at (col 0, mid), roll east
through the middle lane, and are caught by a row of bins (top row). The single
difference is the controller — greedy_bin_phased — which reads the mass of the
ball waiting in each bin's gate cell and only opens the gate when accepting
that specific ball either (a) keeps the bin at or under target (fill phase)
or (b) overshoots by less than OVERSHOOT_TOLERANCE (mop-up phase). Heavier
balls that would overshoot by more than the tolerance are rejected back into
the eastbound lane and recirculate via the W return lane.

OVERSHOOT_TOLERANCE is initialized to the mean of MASS_MIN and MASS_MAX. So
roughly the lighter half of the population is eligible to close a bin once
it has entered the mop-up region.
"""
import numpy as np

# Row 0 is BOTTOM after np.flip.
DIRECTION_MAP = np.flip(np.array([
    # top    — bins on odd cols (5 of them), 'I' separators on even cols and col 10
    ['I', 'B', 'I', 'B', 'I', 'B', 'I', 'B', 'I', 'B', 'I'],
    # mid    — E travel + 'N' gate placeholders on bin cols, 'S' at east drop
    ['E', 'N', 'E', 'N', 'E', 'N', 'E', 'N', 'E', 'N', 'S'],
    # buffer — idle separator between mid (E) and bot (W); 'N' carries the climb through col 0,
    #          'S' carries the drop through col 10
    ['N', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'S'],
    # bot    — return lane 'W', 'N' at west end starts the climb back into the entry cell
    ['N', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
]), 0)

NBALL              = 40
CONTROLLER         = "greedy_bin_phased"
BALL_INIT          = "outside_west_varied"
RESPAWN_STRATEGY   = "west_entry"
RESPAWN_DELAY      = 6.0
MAXSIMULATIONSTEPS = 4000

TARGET_WEIGHT       = 0.06
MASS_MIN            = 0.010
MASS_MAX            = 0.020
OVERSHOOT_TOLERANCE = 0.5 * (MASS_MIN + MASS_MAX)  # 0.015
BALL_RADIUS_SCALE   = 0.5
