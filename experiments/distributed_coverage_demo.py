"""Distributed coverage demo.

Whole-grid distributed bin-coverage via local threshold-triggered diffusion.
Balls are parked off-grid at start (outside_west_varied init) and then
dropped one at a time at the geometric centre of the surface, with
RESPAWN_DELAY seconds between drops -- same delayed-drip pattern used by
the phased controller, just routed to the centre instead of the west entry.

Each edge in the grid runs an independent local rule that lowers the two
pistons along its length when |mass_A - mass_B| is large enough AND the
lighter cell still wants mass AND the heavier cell either also wants mass
or has noticeable excess. Cells equalize toward target; once every cell is
at-or-above TARGET_WEIGHT, no edge meets the firing rule and the system
halts.

No DIRECTION_MAP -- the controller operates on the plain cell grid
uniformly. GRIDSIZEX and GRIDSIZEY are set explicitly.

NBALL is chosen so total injected mass ~= TARGET_WEIGHT * n_cells * (1+margin),
i.e. enough to cover every cell with a small surplus.
"""

GRIDSIZEX = 6
GRIDSIZEY = 6   # 5x5 = 25 cells

CONTROLLER                  = "distributed_coverage"
BALL_INIT                   = "outside_west_varied"
RESPAWN_STRATEGY            = "center"
RESPAWN_DELAY               = 6.0
RESPAWN_REQUIRE_EMPTY_CELL  = False   # allow piling at the centre; diffusion spreads from there
MAXSIMULATIONSTEPS          = 5000

TARGET_WEIGHT     = 0.03
MASS_MIN          = 0.010
MASS_MAX          = 0.020
BALL_RADIUS_SCALE = 0.5

# Inject ~10% more mass than the strict per-cell target sum so boundary cells
# can still close after corners and edges absorb their share.
_n_cells   = (GRIDSIZEX - 1) * (GRIDSIZEY - 1)
_mean_mass = 0.5 * (MASS_MIN + MASS_MAX)
NBALL = int(round(TARGET_WEIGHT * _n_cells * 1.10 / _mean_mass))

# Edge-firing knobs.
DIFFUSION_THRESHOLD_HIGH = MASS_MAX                       # imbalance of ~1 heavy ball -> fire
DIFFUSION_THRESHOLD_LOW  = MASS_MAX                       # no hysteresis for now
DIFFUSION_SAFETY_MARGIN  = 0.5 * (MASS_MIN + MASS_MAX)    # cells within mean_mass of target are protected from donating
DIFFUSION_HOLD_STEPS     = 30                              # 3 seconds at DT=0.1 -- with K=0.2 the rod needs ~15 ticks just to reach LOW, plus time for balls to actually roll across the dip
DIFFUSION_COOLDOWN_STEPS = 10
