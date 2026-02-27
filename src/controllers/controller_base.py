from abc import ABC, abstractmethod
import numpy as np
from src.config import NE, NW, SW, SE


class Controller(ABC):
    """Base class for all surface controllers."""

    def __init__(self, config):
        """
        Initialize the controller.

        Args:
            config: SimConfig instance with simulation parameters
        """
        self.config = config
        self._setup_quadrant_cell_mapping()

    def _setup_quadrant_cell_mapping(self):
        """
        Precompute mapping from piston quadrants to direction map cells.

        For piston at (i, j):
          NE quadrant -> cell (i,   j)
          NW quadrant -> cell (i-1, j)
          SW quadrant -> cell (i-1, j-1)
          SE quadrant -> cell (i,   j-1)
        """
        gx = self.config.GRIDSIZEX
        gy = self.config.GRIDSIZEY
        nx = gx - 1
        ny = gy - 1

        self.quadrant_cell_x = np.full((gx, gy, 4), -1, dtype=int)
        self.quadrant_cell_y = np.full((gx, gy, 4), -1, dtype=int)

        for i in range(gx):
            for j in range(gy):
                if i < nx and j < ny:
                    self.quadrant_cell_x[i, j, NE] = i
                    self.quadrant_cell_y[i, j, NE] = j
                if i > 0 and j < ny:
                    self.quadrant_cell_x[i, j, NW] = i - 1
                    self.quadrant_cell_y[i, j, NW] = j
                if i > 0 and j > 0:
                    self.quadrant_cell_x[i, j, SW] = i - 1
                    self.quadrant_cell_y[i, j, SW] = j - 1
                if i < nx and j > 0:
                    self.quadrant_cell_x[i, j, SE] = i
                    self.quadrant_cell_y[i, j, SE] = j - 1

    def _get_direction_for_quadrant(self, i, j, quadrant):
        """Get the direction string from the map for a piston's quadrant."""
        cx = self.quadrant_cell_x[i, j, quadrant]
        cy = self.quadrant_cell_y[i, j, quadrant]
        if cx < 0 or cy < 0:
            return 'I'
        return self.direction_map[cy, cx]

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
