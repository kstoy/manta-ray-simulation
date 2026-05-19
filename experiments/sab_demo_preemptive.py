"""SAB demo — priority-preemptive controller with custom direction map on a 14x8 grid."""
import numpy as np

# Row 0 is BOTTOM, last row is TOP.
# Directions: N S E W I (idle). Priority: "NE" = prefer N, fallback E.
DIRECTION_MAP = np.flip(np.array([
    ['S',  'SW', 'SW', 'SW','W', 'SW','SW',   'SW','W', 'SW', 'SW',   'W', "W"],  # row 7
    ['S',  'S',  'I',  'I', 'N', 'S', 'I',    'I', 'N', 'S',  'I',    'N', "N"],  # row 6
    ['S',  'I',  'E',  'E', 'N', 'I', 'NEWS', 'I', 'N', 'I',  'NEWS', 'I', "WN"],  # row 5
    ['ES', 'E',  'I',  'W', 'N', 'E', 'I',    'N', 'N', 'E',  'I',    'E', "N"],  # row 4
    ['S',  'W',  'W',  'N', 'N', 'N', 'S',    'N', 'N', 'N',  'NEWS', 'I', "WN"],  # row 3
    ['S',  'I',  'I',  'N', 'N', 'N', 'S',    'N', 'N', 'N',  'I',    'W', "N"],  # row 2
    ['E',  'NE', 'NE', 'NE','E', 'NE', 'E',  'NE', 'E', 'NE', 'NE',   'E', "N"],  # row 1
]), 0)

NBALL              = 33
BALL_RADIUS        = 0.1
CONTROLLER         = "priority_preemptive"
BALL_INIT          = "perimeter"
RESPAWN_STRATEGY   = "perimeter"
RESPAWN_DELAY      = 0.0
MAXSIMULATIONSTEPS = 3000
