"""ITU demo — priority controller with custom direction map on a 14x8 grid."""
import numpy as np

# Row 0 is BOTTOM, last row is TOP.
# Directions: N S E W I (idle). Priority: "NE" = prefer N, fallback E.
DIRECTION_MAP = np.flip(np.array([
    ['S', 'W', 'SW','SW','SW','W', 'SW','SW','SW','W', 'W',  'SW','W', 'W', "W"],  # row 7
    ['S', 'W', 'S', 'I', 'I', 'N', 'S', 'I', 'S', 'N', 'I',  'W', 'N', 'N', "N"],  # row 6
    ['S', 'W', 'S', 'E', 'E', 'N', 'S', 'I', 'S', 'N', 'I',  'I', 'I', 'E', "N"],  # row 5
    ['S', 'W', 'I', 'W', 'W', 'N', 'SE','I', 'S', 'N', 'NE', 'I', 'N', 'E', "N"],  # row 4
    ['S', 'W', 'W', 'W', 'N', 'N', 'S', 'S', 'S', 'N', 'N',  'I', 'N', 'E', "N"],  # row 3
    ['S', 'S', 'I', 'I', 'N', 'N', 'I', 'S', 'I', 'N', 'NE', 'I', 'N', 'E', "N"],  # row 2
    ['E', 'E', 'NE','NE','NE','E', 'E', 'E', 'E', 'E', 'NE', 'NE','NE','E', "N"],  # row 1 
]), 0)

NBALL              = 35
BALL_RADIUS        = 0.1
CONTROLLER         = "priority"
BALL_INIT          = "perimeter"
RESPAWN_STRATEGY   = "southwest"
RESPAWN_DELAY      = 0.0
MAXSIMULATIONSTEPS = 3000
