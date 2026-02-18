import numpy as np

NBALL = 20

# 9x9 direction map pointing toward center. Row 0 is BOTTOM, last row is TOP.
# Directions: N S E W I (idle).
DIRECTION_MAP = np.flip(np.array([
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],  # top
    ['E', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'W'],
    ['E', 'E', 'S', 'S', 'S', 'S', 'S', 'W', 'W'],
    ['E', 'E', 'E', 'S', 'S', 'S', 'W', 'W', 'W'],
    ['E', 'E', 'E', 'E', 'I', 'W', 'W', 'W', 'W'],  # center
    ['E', 'E', 'E', 'N', 'N', 'N', 'W', 'W', 'W'],
    ['E', 'E', 'N', 'N', 'N', 'N', 'N', 'W', 'W'],
    ['E', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'W'],
    ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],  # bottom
]), 0)

CONTROLLER         = "nonblocking"
BALL_INIT          = "outside_rectangle"
RESPAWN            = True
MAXSIMULATIONSTEPS = 2750
