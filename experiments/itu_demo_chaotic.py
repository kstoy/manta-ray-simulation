"""ITU demo (chaotic) — priority controller with custom direction map on a 16x10 grid."""
import numpy as np

# Row 0 is BOTTOM, last row is TOP.
# Directions: N S E W I (idle). Priority: "NE" = prefer N, fallback E.
# Outer border drives CCW circulation: bottom=E, right=N, top=W, left=S.
DIRECTION_MAP = np.flip(np.array([
    ['S', "W",  'W', 'W',  'W', 'W', 'W', 'W',  'W',  'W', 'W', 'W', 'W', 'W', 'W', 'W'],  # row 8 (top border)
    ['S', "W",  'I', 'I',  'N', 'I', 'I', 'I',  'I',  'N', 'I', 'I', 'N', 'I', 'I', 'N'],  # row 6
    ['S', "W",  'N', 'N',  'N', 'N', 'WN','EN', 'N',  'N', 'N', 'N', 'N', 'N', 'N', 'N'],  # row 5
    ['S', "W",  'N', 'N',  'N', 'W', 'N', 'N',  'E', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],  # row 4
    ['S', "W",  'N', 'N',  'N', 'N', 'N', 'N',  'E', 'N', 'N', 'N', 'I', 'N', 'N', 'N'],  # row 3
    ['S', "W",  'N', 'N',  'N', 'N', 'N', 'N',  'E',  'N', 'N', 'N', 'N', 'N', 'N', 'N'],  # row 2
    ['E', "E",  'NE','NE', 'E', 'E', 'NE','NE', 'E',  'E', 'NE','NE','NE','NE','NE','N'],  # row 1
]), 0)

NBALL              = 46
CONTROLLER         = "priority_delayed"
BALL_INIT          = "outside_rectangle"
RESPAWN_STRATEGY   = "random"
RESPAWN_DELAY      = 0.0
MAXSIMULATIONSTEPS = 2500
