"""Weight-based sorting demo.

7x3 grid. Balls enter from the west at the middle row, hit a sorter cell at
(1, 1), and get routed N / E / S depending on their mass. All three branches
park at the east column.

Masses are uniform random in [0.005, 0.020] kg. The sorter splits that range
into thirds:
  m < 0.010           -> light  -> S
  0.010 <= m < 0.015  -> medium -> E
  m >= 0.020          -> heavy  -> N
"""
import numpy as np

# Row 0 is BOTTOM after np.flip. Sorter at cell (1, 1); its placeholder 'I'
# is overwritten at runtime by the latched direction.
DIRECTION_MAP = np.flip(np.array([
    ['I', 'E', 'E', 'E', 'E', 'E', 'I'],  # row 2 (top)    — N branch parks at east column
    ['E', 'I', 'E', 'E', 'E', 'E', 'I'],  # row 1 (mid)    — entry, sorter, E branch parks
    ['I', 'E', 'E', 'E', 'E', 'E', 'I'],  # row 0 (bottom) — S branch parks at east column
]), 0)

SORTER_MAP = {
    (1, 1): {
        'thresholds': [0.010, 0.015],
        'directions': ['S', 'E', 'N'],   # light -> S, medium -> E, heavy -> N
    },
}

NBALL              = 9
CONTROLLER         = "blocking_weighted"
BALL_INIT          = "outside_west_varied"
RESPAWN_STRATEGY   = "west_entry"
RESPAWN_DELAY      = 8.0
MAXSIMULATIONSTEPS = 3000
