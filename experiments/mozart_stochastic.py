"""MOZART demo — stochastic-assembly controller.

Forms the BIT_MAP shape by drift+diffusion self-assembly instead of directed
routing: per-cell stay-probability (h-pinning + penalty-coupling) decides where
balls rest, a CCW-circulation drift (plus weak attract) keeps the gas moving,
and enclosed holes are auto-forbidden.  See
src/controllers/controller_stochastic_assembly.py for the model and SA_* knobs
in src/config.py for tuning / annealing.

The BIT_MAP is passed in the DIRECTION_MAP slot; the controller reads it as the
target pattern.  Use even row/column counts so there is no degenerate centre.
"""
import numpy as np

# Row 0 is BOTTOM after the flip. '1' marks a cell of the target shape.
#         |------- M  O  Z -------|  |------- A -------|  |------ R ------|  |------ T ------|
BIT_MAP = np.flip(np.array([
    ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],  # row 8 (top)
    ['0', '1', '0', '0', '1', '0', '0', '1', '0', '0', '1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1', '0'],  # row 7
    ['0', '1', '1', '1', '1', '0', '1', '0', '1', '0', '0', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '0', '1', '0', '0'],  # row 6
    ['0', '1', '0', '0', '1', '0', '1', '0', '1', '0', '0', '1', '0', '0', '1', '1', '1', '0', '1', '1', '1', '0', '0', '1', '0', '0'],  # row 5
    ['0', '1', '0', '0', '1', '0', '1', '0', '1', '0', '0', '1', '0', '0', '1', '0', '1', '0', '1', '1', '0', '0', '0', '1', '0', '0'],  # row 4
    ['0', '1', '0', '0', '1', '0', '1', '0', '1', '0', '1', '0', '0', '0', '1', '0', '1', '0', '1', '0', '1', '0', '0', '1', '0', '0'],  # row 3
    ['0', '1', '0', '0', '1', '0', '0', '1', '0', '0', '1', '1', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '0', '1', '0', '0'],  # row 2
    ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],  # row 1 (bottom)
]), 0)

# The controller reads the pattern from the DIRECTION_MAP slot.
DIRECTION_MAP = BIT_MAP

CONTROLLER         = "stochastic_assembly"
# Sweep result: high pinning makes placed balls stubborn (4x fewer strays, lowest
# jitter); SA_P_OFF stays at its default 2 since off-pattern balls are already
# maximally eager to move at the cold end-temperature.
SA_H               = 10.0
# One ball per '1', plus a small mobile-gas surplus to feed the growth front.
NBALL              = int((BIT_MAP == '1').sum())
BALL_RADIUS        = 0.1
BALL_INIT          = "random"
RESPAWN_STRATEGY   = "perimeter"
RESPAWN_DELAY      = 0.0
# Stochastic assembly mixes slower than directed routing — give it more steps.
MAXSIMULATIONSTEPS = 8000
