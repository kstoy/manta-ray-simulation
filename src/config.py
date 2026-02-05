from dataclasses import dataclass


@dataclass
class SimConfig:
    # Grid geometry
    GRIDSIZEX: int = 10
    GRIDSIZEY: int = 10

    # distance between rods, fabric side length factor ( fabric side length = D*LF )
    D: float = 1.0
    LF: float = 1.45


    # Simulation
    MAXSIMULATIONSTEPS: int = 2750
    DT: float = 0.1
    MAXCOEFF: int = 5

    # Ball initialization
    BALL_INIT: str = "outside_rectangle"  # Ball init: "grid_uniform", "random", "center_cluster"
    RESPAWN: bool = True  # Reset out-of-bounds balls to NW corner

    # Control
    CONTROLLER: str = "direction_map_single" # "adaptive_threshold"  # Controller type: "square_push", "square_pull", "weight_sort", etc.
    K: float = 0.2
    TARGET_WEIGHT: float = 0.04
    P: float = 2.0

    # Visualization
    TRIANGLES: int = 9
    EXPLODE: float = 1.0
    SIGMA: float = 0.01

    NBALL = 20