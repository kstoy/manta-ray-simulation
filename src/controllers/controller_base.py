from abc import ABC, abstractmethod
import numpy as np


class Controller(ABC):
    """Base class for all surface controllers."""

    def __init__(self, config):
        """
        Initialize the controller.

        Args:
            config: SimConfig instance with simulation parameters
        """
        self.config = config

    @abstractmethod
    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        """
        Compute desired rod height for position (i, j).

        Args:
            i: Rod x-index
            j: Rod y-index
            timestep: Current simulation timestep
            sensors: Sensor readings at this rod [NE, NW, SW, SE]

        Returns:
            Desired rod height (typically 0.5 to 1.5)
        """
        pass

    def update_all(self, timestep: int, sensors: np.ndarray) -> np.ndarray:
        """
        Compute desired rod heights for all positions at once (vectorized).

        Override this method in subclasses for better performance.
        Default implementation falls back to per-rod update() calls.

        Args:
            timestep: Current simulation timestep
            sensors: Sensor array of shape (GRIDSIZEX, GRIDSIZEY, 4)

        Returns:
            Array of desired heights with shape (GRIDSIZEX, GRIDSIZEY)
        """
        # Fallback: use nested loops calling update()
        desired = np.empty((self.config.GRIDSIZEX, self.config.GRIDSIZEY))
        for i in range(self.config.GRIDSIZEX):
            for j in range(self.config.GRIDSIZEY):
                desired[i, j] = self.update(i, j, timestep, sensors[i, j])
        return desired
