"""
Simple Direction Map Controller

Similar to DirectionMapController but does not check if the destination is free.
Just checks if there's a ball in a quadrant that has a valid outgoing direction.
"""

import numpy as np
from src.controllers.controller_base import Controller
from src.constants import NE, NW, SW, SE


class DirectionMapControllerMulti(Controller):
    """Controller that uses a direction map without checking destination availability."""

    # Direction constants
    DIR_N = 'N'
    DIR_S = 'S'
    DIR_E = 'E'
    DIR_W = 'W'
    DIR_I = 'I'  # Idle/no action

    # Valid source quadrants for each direction
    # S: can move from NE or NW (northern quadrants moving south)
    # N: can move from SE or SW (southern quadrants moving north)
    # E: can move from NW or SW (western quadrants moving east)
    # W: can move from NE or SE (eastern quadrants moving west)
    VALID_SOURCES = {
        'S': {NE, NW},
        'N': {SE, SW},
        'E': {NW, SW},
        'W': {NE, SE},
        'I': set(),
    }

    def __init__(self, config, direction_map=None):
        """
        Initialize the simple direction map controller.

        Args:
            config: SimConfig instance
            direction_map: 2D array of direction chars ('N','S','E','W','I')
                          Shape should be (GRIDSIZEY-1, GRIDSIZEX-1) for cells.
                          If None, creates a default inward-spiral pattern.
        """
        super().__init__(config)

        if direction_map is None:
            self.direction_map = self._create_default_direction_map()
        else:
            self.direction_map = np.array(direction_map)

        # Precompute cell-to-quadrant mapping
        self._setup_quadrant_cell_mapping()

        # Print the map once at initialization
        self.print_direction_map()

    def _create_default_direction_map(self):
        """
        Create a default direction map that guides balls toward center.
        Returns array indexed as [y, x] with shape (GRIDSIZEY-1, GRIDSIZEX-1).
        """
        nx = self.config.GRIDSIZEX - 1
        ny = self.config.GRIDSIZEY - 1

        direction_map = np.full((ny, nx), self.DIR_I, dtype='U1')

        cx = nx / 2.0
        cy = ny / 2.0

        for j in range(ny):
            for i in range(nx):
                dx = (i + 0.5) - cx
                dy = (j + 0.5) - cy

                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    direction_map[j, i] = self.DIR_I
                elif abs(dx) > abs(dy):
                    direction_map[j, i] = self.DIR_W if dx > 0 else self.DIR_E
                else:
                    direction_map[j, i] = self.DIR_S if dy > 0 else self.DIR_N

        return direction_map

    def _setup_quadrant_cell_mapping(self):
        """Setup mapping from piston quadrants to direction map cells."""
        gx = self.config.GRIDSIZEX
        gy = self.config.GRIDSIZEY
        nx = gx - 1
        ny = gy - 1

        self.quadrant_cell_x = np.full((gx, gy, 4), -1, dtype=int)
        self.quadrant_cell_y = np.full((gx, gy, 4), -1, dtype=int)

        for i in range(gx):
            for j in range(gy):
                # NE quadrant -> cell (i, j)
                if i < nx and j < ny:
                    self.quadrant_cell_x[i, j, NE] = i
                    self.quadrant_cell_y[i, j, NE] = j

                # NW quadrant -> cell (i-1, j)
                if i > 0 and j < ny:
                    self.quadrant_cell_x[i, j, NW] = i - 1
                    self.quadrant_cell_y[i, j, NW] = j

                # SW quadrant -> cell (i-1, j-1)
                if i > 0 and j > 0:
                    self.quadrant_cell_x[i, j, SW] = i - 1
                    self.quadrant_cell_y[i, j, SW] = j - 1

                # SE quadrant -> cell (i, j-1)
                if i < nx and j > 0:
                    self.quadrant_cell_x[i, j, SE] = i
                    self.quadrant_cell_y[i, j, SE] = j - 1

    def _get_direction_for_quadrant(self, i, j, quadrant):
        """Get the direction from the map for a piston's quadrant."""
        cx = self.quadrant_cell_x[i, j, quadrant]
        cy = self.quadrant_cell_y[i, j, quadrant]
        if cx < 0 or cy < 0:
            return self.DIR_I
        return self.direction_map[cy, cx]

    def _check_quadrant_action(self, sensors, quadrant, direction):
        """
        Check if a quadrant's direction condition triggers a lower action.
        Only checks if there's a ball in a valid source quadrant for the direction.
        """
        if direction == self.DIR_I:
            return False

        # Check if this quadrant is a valid source for this direction
        valid_sources = self.VALID_SOURCES.get(direction, set())
        if quadrant not in valid_sources:
            return False

        # Check if there's a ball in this quadrant
        return sensors[quadrant] > 0

    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        """
        Compute desired rod height for position (i, j).
        Lowers piston if any quadrant has a ball with valid outgoing direction.
        """
        for quadrant in [NE, NW, SW, SE]:
            direction = self._get_direction_for_quadrant(i, j, quadrant)
            if self._check_quadrant_action(sensors, quadrant, direction):
                return 0.5  # Lower the piston

        return 1.5  # Raise the piston

    def set_direction_map(self, direction_map):
        """Update the direction map at runtime."""
        self.direction_map = np.array(direction_map)

    def get_direction_map(self):
        """Return the current direction map."""
        return self.direction_map.copy()

    def print_direction_map(self):
        """Print the direction map in a readable format."""
        print("Direction Map (Multi):")
        print("  " + " ".join(f"{i}" for i in range(self.direction_map.shape[1])))
        for j in range(self.direction_map.shape[0] - 1, -1, -1):
            row = " ".join(self.direction_map[j, :])
            print(f"{j} {row}")
