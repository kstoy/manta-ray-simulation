"""
Priority Direction Map Controller

Uses a 2D direction map where each cell contains a priority list of directions.
For example, "NW" means: try North first, if blocked try West.

Directions are tried in order until one succeeds (destination is free) or all fail.
"""

import numpy as np
from src.controllers.controller_base import Controller
from src.constants import NE, NW, SW, SE


class DirectionMapControllerPriority(Controller):
    """Controller that uses a direction map with priority fallback directions."""

    # Direction constants
    DIR_N = 'N'
    DIR_S = 'S'
    DIR_E = 'E'
    DIR_W = 'W'
    DIR_I = 'I'  # Idle/no action

    # Mapping from quadrant to destination quadrant for each direction
    # Moving South: NE->SE, NW->SW
    SOUTH_OF = {NE: SE, NW: SW, SE: None, SW: None}
    # Moving North: SE->NE, SW->NW
    NORTH_OF = {SE: NE, SW: NW, NE: None, NW: None}
    # Moving East: NW->NE, SW->SE
    EAST_OF = {NW: NE, SW: SE, NE: None, SE: None}
    # Moving West: NE->NW, SE->SW
    WEST_OF = {NE: NW, SE: SW, NW: None, SW: None}

    DIRECTION_MAPS = {
        'N': NORTH_OF,
        'S': SOUTH_OF,
        'E': EAST_OF,
        'W': WEST_OF,
    }

    def __init__(self, config, direction_map=None):
        """
        Initialize the priority direction map controller.

        Args:
            config: SimConfig instance
            direction_map: 2D array of direction strings (e.g., 'N', 'NW', 'SEW', 'I')
                          Each string is a priority list: first char tried first.
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
        Create a default direction map with priority fallbacks toward center.
        Returns array indexed as [y, x] with shape (GRIDSIZEY-1, GRIDSIZEX-1).
        """
        nx = self.config.GRIDSIZEX - 1
        ny = self.config.GRIDSIZEY - 1

        direction_map = np.full((ny, nx), self.DIR_I, dtype='U4')

        cx = nx / 2.0
        cy = ny / 2.0

        for j in range(ny):
            for i in range(nx):
                dx = (i + 0.5) - cx
                dy = (j + 0.5) - cy

                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    # At center - idle
                    direction_map[j, i] = self.DIR_I
                else:
                    # Primary direction toward center, secondary as fallback
                    primary = ''
                    secondary = ''

                    if abs(dx) > abs(dy):
                        # Primarily horizontal
                        primary = 'W' if dx > 0 else 'E'
                        secondary = 'S' if dy > 0 else 'N'
                    else:
                        # Primarily vertical
                        primary = 'S' if dy > 0 else 'N'
                        secondary = 'W' if dx > 0 else 'E'

                    direction_map[j, i] = primary + secondary

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

    def _get_directions_for_quadrant(self, i, j, quadrant):
        """Get the priority direction string from the map for a piston's quadrant."""
        cx = self.quadrant_cell_x[i, j, quadrant]
        cy = self.quadrant_cell_y[i, j, quadrant]
        if cx < 0 or cy < 0:
            return self.DIR_I
        return self.direction_map[cy, cx]

    def _check_quadrant_action(self, sensors, quadrant, directions):
        """
        Check if a quadrant's direction conditions trigger a lower action.
        Tries each direction in priority order until one succeeds.

        Args:
            sensors: Sensor readings [NE, NW, SW, SE] for this piston
            quadrant: Which quadrant we're checking (NE, NW, SW, SE)
            directions: Priority string of directions (e.g., "NW" means try N, then W)

        Returns:
            True if piston should be lowered, False otherwise
        """
        if directions == self.DIR_I or not directions:
            return False

        # Check if there's a ball in this quadrant first
        if sensors[quadrant] == 0:
            return False

        # Try each direction in priority order
        for direction in directions:
            if direction not in self.DIRECTION_MAPS:
                continue

            dest_map = self.DIRECTION_MAPS[direction]
            dest_quadrant = dest_map.get(quadrant)

            # Skip if no valid destination for this direction from this quadrant
            if dest_quadrant is None:
                continue

            # Check if destination is free
            if sensors[dest_quadrant] == 0:
                # Found a valid move - lower the piston
                return True

        # No valid direction found
        return False

    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        """
        Compute desired rod height for position (i, j).
        Checks all 4 quadrants and lowers piston if any direction condition is met.
        """
        for quadrant in [NE, NW, SW, SE]:
            directions = self._get_directions_for_quadrant(i, j, quadrant)
            if self._check_quadrant_action(sensors, quadrant, directions):
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
        print("Direction Map (Priority):")
        # Find max width for formatting
        max_width = max(len(str(d)) for row in self.direction_map for d in row)
        max_width = max(max_width, 2)

        header = "  " + " ".join(f"{i:>{max_width}}" for i in range(self.direction_map.shape[1]))
        print(header)
        for j in range(self.direction_map.shape[0] - 1, -1, -1):
            row = " ".join(f"{self.direction_map[j, i]:>{max_width}}" for i in range(self.direction_map.shape[1]))
            print(f"{j} {row}")
