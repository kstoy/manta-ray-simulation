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

    # Ball initialization
    BALL_INIT: str = "outside_rectangle"  # Ball init: "grid_uniform", "random", "center_cluster"
    RESPAWN_STRATEGY: object = "random"  # Respawn strategy name ("random", "grid_uniform", etc.), or None to disable
    RESPAWN_DELAY: float = 0.0  # Seconds a ball must be OOB before respawning (also used as global cooldown)

    # Control
    CONTROLLER: object = "blocking"  # str name ("blocking", "nonblocking", "priority") or callable(config)
    K: float = 0.2
    TARGET_WEIGHT: float = 0.04
    P: float = 2.0

    # Visualization
    TRIANGLES: int = 9
    EXPLODE: float = 1.0
    SIGMA: float = 0.01

    NBALL: int = 20
    BALL_RADIUS: float = 0.05