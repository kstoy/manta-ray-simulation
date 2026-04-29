"""ITU demo — priority controller with custom direction map on a 14x8 grid."""
import numpy as np

# Row 0 is BOTTOM, last row is TOP.
# Directions: N S E W I (idle). Priority: "NE" = prefer N, fallback E.
DIRECTION_MAP = np.flip(np.array([
    ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I',  'I', 'I'],  # row 7
    ['I', 'I', 'I', 'I', 'W', 'E', 'I', 'I', 'I', 'I', 'I',  'I', 'I'],  # row 6
    ['N', 'N', 'I', 'I', 'WN','NE','I', 'I', 'N', 'N', 'I',  'N', 'N'],  # row 5
    ['N', 'N', 'I', 'I', 'N', 'N', 'I', 'I', 'N', 'N', 'I',  'N', 'N'],  # row 4
    ['N', 'N', 'I', 'I', 'N', 'N', 'I', 'I', 'N', 'N', 'I',  'N', 'N'],  # row 3
    ['N', 'N', 'I', 'I', 'N', 'N', 'I', 'I', 'N', 'N', 'N',  'N', 'N'],  # row 2
    ['NE','NE','E', 'E', 'NE','NE','E', 'E', 'NE','NE','NE', 'NE','N'],  # row 1 
]), 0)

NBALL              = 46
CONTROLLER         = "priority"
BALL_INIT          = "outside_rectangle"
RESPAWN_STRATEGY   = "southwest"
RESPAWN_DELAY      = 5.0
MAXSIMULATIONSTEPS = 2750
