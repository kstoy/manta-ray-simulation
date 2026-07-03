"""
Convergence test for the random-pair controller.

Setup: a small rod grid, one ball per cell on average, balls placed uniformly
at random across the surface.  The controller repeatedly picks a random pair
of adjacent rods, lowers them to create a valley, and raises them again.
Each event roughly averages the mass in the two adjacent cells, so the
hypothesis is that the per-cell ball count converges toward 1.

Respawn is disabled so any boundary losses are visible rather than masked.
"""

GRIDSIZEX = 12
GRIDSIZEY = 12

NBALL = (GRIDSIZEX - 1) * (GRIDSIZEY - 1)   # 121 cells, 121 balls (1 per cell on average)

CONTROLLER             = "random_pair"
BALL_INIT              = "random"           # uniform random positions on the surface
RESPAWN_STRATEGY       = None               # disabled — measure conservation
MAXSIMULATIONSTEPS     = 5000

# Hold time per fired edge (longer = balls have more time to settle in the trough).
RANDOM_PAIR_HOLD_STEPS = 40
