from dataclasses import dataclass

# Sensor quadrant direction indices
NE = 0
NW = 1
SW = 2
SE = 3


@dataclass
class SimConfig:
    # Grid geometry
    GRIDSIZEX: int = 10
    GRIDSIZEY: int = 10

    # Distance between adjacent rods (m) and side length of the fabric panel
    # spanning each rod cell (m).  D_FABRIC > D_RODS gives slack so the fabric sags.
    D_RODS: float = 0.5
    D_FABRIC: float = 0.6

    # Rod travel limits (m).  Controllers command rods to LOW_HEIGHT (lower) or
    # HIGH_HEIGHT (raise); rods rest at the midpoint at startup.
    LOW_HEIGHT: float = 0.7
    HIGH_HEIGHT: float = 1.0


    # Simulation
    MAXSIMULATIONSTEPS: int = 2750
    DT: float = 0.1
    MAXCOEFF: int = 5
    # Master RNG seed for stochastic ball init / respawn (e.g. perimeter_random).
    # None -> non-deterministic; set an int for reproducible-but-distinct runs.
    SEED: object = None

    # Ball initialization
    BALL_INIT: str = "outside_rectangle"  # Ball init: "grid_uniform", "random", "center_cluster"
    RESPAWN_STRATEGY: object = "random"  # Respawn strategy name ("random", "grid_uniform", etc.), or None to disable
    RESPAWN_DELAY: float = 0.0  # Seconds a ball must be OOB before respawning (also used as global cooldown)
    RESPAWN_REQUIRE_EMPTY_CELL: bool = True  # If True, skip respawn while another ball already occupies the spawn cell (default behaviour); set False to allow piling at a fixed drop point (e.g. distributed_coverage)
    MASS_MIN: float = 0.005  # Lower bound for uniform random ball mass (kg); used by outside_west_varied
    MASS_MAX: float = 0.020  # Upper bound for uniform random ball mass (kg); used by outside_west_varied
    BALL_RADIUS_SCALE: float = 1.0  # Multiplier on the radius computed from mass; used by outside_west_varied

    # Control
    CONTROLLER: object = "blocking"  # str name ("blocking", "nonblocking", "priority") or callable(config)
    K: float = 0.2
    TARGET_WEIGHT: float = 0.04
    OVERSHOOT_TOLERANCE: float = 0.015  # greedy_bin_phased: in mop-up phase, accept incoming balls whose mass is <= this
    # distributed_coverage controller
    DIFFUSION_THRESHOLD_HIGH: float = 0.000   # |mass_A - mass_B| must exceed this to trigger an edge firing
    DIFFUSION_THRESHOLD_LOW:  float = 0.000   # reserved for hysteresis (re-fire only after diff drops below LOW); set == HIGH for now
    DIFFUSION_SAFETY_MARGIN:  float = 0.000   # a cell at or above target only donates when its mass exceeds target + this
    DIFFUSION_HOLD_STEPS:     int = 10        # ticks an edge stays LOW once fired (~1 second at DT=0.1)
    DIFFUSION_COOLDOWN_STEPS: int = 10        # ticks an edge is locked HIGH after the hold expires, before it can re-fire
    P: float = 2.0

    # random_pair controller: timesteps each fired edge holds both rods at LOW_HEIGHT,
    # timesteps after the LOW phase ends during which the edge's rods stay reserved
    # (rising back to HIGH), and timesteps after the reservation is released during
    # which the same edge cannot re-fire (forces rotation across the grid).
    RANDOM_PAIR_HOLD_STEPS: int = 40
    RANDOM_PAIR_COOLDOWN_STEPS: int = 25
    RANDOM_PAIR_REFRACTORY_STEPS: int = 200

    # stochastic_assembly controller (drift + diffusion self-assembly).
    # Placement: P_stay = sigmoid(beta_temp * (h*[pattern] - q*misplaced_neighbours
    # - p*[off-pattern])).  Movement: softmax over neighbours biased by
    # gamma*CCW_circulation + alpha*attract_gradient, sharpness beta_drift.
    # beta_temp / beta_drift / q are annealed start->end over the run.
    SA_H: float = 3.0               # pinning field: pattern-cell baseline stability
    SA_P_OFF: float = 2.0           # off-pattern instability (keeps the gas mobile)
    SA_BETA_START: float = 0.5      # placement inverse-temperature (hot early)
    SA_BETA_END: float = 4.0        # cold late -> lock the pattern
    SA_BETADRIFT_START: float = 3.0 # directed drift early (fill fast)
    SA_BETADRIFT_END: float = 0.5   # diffusive late (self-heal)
    SA_Q_START: float = 2.0         # penalty: porous walls early (clear traps)
    SA_Q_END: float = 0.5           # rigid walls late (hold the shape)
    SA_ALPHA: float = 0.5           # attract-gradient drift weight (keep small)
    SA_GAMMA: float = 1.0           # CCW-circulation drift weight
    SA_ESCAPE_STRENGTH: float = 4.0 # outward bias inside forbidden holes
    SA_PRESENT_FRACTION: float = 0.5   # cell occupied when its mass exceeds this fraction of one ball
    SA_TRANSFER_FRACTION: float = 0.5  # end an eject once the destination holds this fraction of one ball
    SA_SEED: int = 0                # RNG seed
    SA_FORBID_HOLES: bool = True    # mark border-unreachable 0-regions as forbidden
    # Cell state-machine timing (steps).  REST_HOLD must exceed the rod settle
    # time under gain K (~log(eps)/log(1-K); K=0.2 -> ~20 steps).
    SA_REST_HOLD_STEPS: int = 25    # catch-and-rest hold: rods reach full height, ball settles
    SA_CATCH_STEPS: int = 10        # far-edge hand-off pulse on arrival (0 disables it)
    SA_EJECT_STEPS: int = 24        # edge-lower hold while a ball crosses to the neighbour

    # Visualization
    TRIANGLES: int = 9
    EXPLODE: float = 1.0
    SIGMA: float = 0.01

    NBALL: int = 20
    BALL_RADIUS: float = 0.05