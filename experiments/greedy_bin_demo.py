"""Greedy bin-filling baseline.

4x11 cell grid. Balls drop in at (col 0, mid), roll east through the middle
lane, and are caught by the first open bin (top row). Bins are placed on
odd columns separated by idle cells; col 0 is the entry and col 10 is the
east drop. Bins start in state 'B' (active/open) and latch to 'I' once
their accumulated mass reaches TARGET_WEIGHT.

An idle buffer row sits between the eastbound mid lane and the westbound
return lane, so the opposing flows never share a piston — preventing the
E-vs-W deadlock that would otherwise occur at every interior piston row j=1.

The middle row's gate cells (under bin columns) are placeholders 'N' — the
controller flips them between 'N' (gate open while the bin is 'B') and 'E'
(gate closed once the bin is 'I'). The north piston row mirrors the row
below at piston columns adjacent to a 'B' cell, and stays HIGH otherwise.

Strict first-fit: bins fill west-to-east. Used as a baseline for comparison
against more sophisticated local strategies.
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
CONTROLLER         = "greedy_bin"
BALL_INIT          = "outside_west_varied"
RESPAWN_STRATEGY   = "west_entry"
RESPAWN_DELAY      = 6.0
MAXSIMULATIONSTEPS = 4000

TARGET_WEIGHT      = 0.06
MASS_MIN           = 0.010
MASS_MAX           = 0.020
BALL_RADIUS_SCALE  = 0.5
