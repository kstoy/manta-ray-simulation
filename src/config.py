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
    MAXSIMULATIONSTEPS: int = 200
    DT: float = 0.1
    MAXCOEFF: int = 5

    # Control
    CONTROLLER: str = "square_push"  # Controller type: "square_push", "square_pull", "weight_sort", etc.
    K: float = 0.2
    TARGET_WEIGHT: float = 0.04
    P: float = 2.0

    # Visualization
    TRIANGLES: int = 9
    EXPLODE: float = 1.0
    SIGMA: float = 0.01

    @property
    def NBALL(self) -> int:
        return (self.GRIDSIZEX - 1)*(self.GRIDSIZEY - 1)
