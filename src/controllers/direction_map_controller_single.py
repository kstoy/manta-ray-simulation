"""
Direction Map Controller

Uses a 2D direction map where each cell contains a direction literal (E, N, W, S, I).
For each piston, inspects its 4 surrounding quadrants and based on the direction
in each quadrant, checks sensor readings to decide whether to lower the piston.

Logic for direction "S" in quadrant Q:
  - Check if there's a ball in Q (sensor reading above threshold)
  - Check if there's no ball in the quadrant to the South of Q
  - If both true: lower piston to create slope toward South

Similar logic applies for N, E, W directions.
I (idle) means no action for that quadrant.
"""

import numpy as np
from src.controllers.controller_base import Controller
from src.constants import NE, NW, SW, SE


class DirectionMapControllerSingle(Controller):
    """Controller that uses a direction map to guide ball movement (checks destination is free)."""

    # Direction constants
    DIR_N = 'N'
    DIR_S = 'S'
    DIR_E = 'E'
    DIR_W = 'W'
    DIR_I = 'I'  # Idle/no action

    # Mapping from quadrant to "south of" quadrant (for S direction)
    # Moving South: NE->SE, NW->SW
    SOUTH_OF = {NE: SE, NW: SW, SE: None, SW: None}
    # Moving North: SE->NE, SW->NW
    NORTH_OF = {SE: NE, SW: NW, NE: None, NW: None}
    # Moving East: NW->NE, SW->SE
    EAST_OF = {NW: NE, SW: SE, NE: None, SE: None}
    # Moving West: NE->NW, SE->SW
    WEST_OF = {NE: NW, SE: SW, NW: None, SW: None}

    def __init__(self, config, direction_map=None):
        """
        Initialize the direction map controller.

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

        # Precompute cell-to-quadrant mapping for vectorized updates
        self._setup_quadrant_cell_mapping()

        # Print the map once at initialization
        self.print_direction_map()

    def _create_default_direction_map(self):
        """
        Create a default direction map that guides balls toward center.
        Returns array indexed as [y, x] with shape (GRIDSIZEY-1, GRIDSIZEX-1).
        """
        nx = self.config.GRIDSIZEX - 1  # Number of cells in x
        ny = self.config.GRIDSIZEY - 1  # Number of cells in y

        direction_map = np.full((ny, nx), self.DIR_I, dtype='U1')

        cx = nx / 2.0  # Center x
        cy = ny / 2.0  # Center y

        for j in range(ny):
            for i in range(nx):
                # Distance from center
                dx = (i + 0.5) - cx
                dy = (j + 0.5) - cy

                # Determine primary direction toward center
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    # At center - idle
                    direction_map[j, i] = self.DIR_I
                elif abs(dx) > abs(dy):
                    # Move horizontally
                    direction_map[j, i] = self.DIR_W if dx > 0 else self.DIR_E
                else:
                    # Move vertically
                    direction_map[j, i] = self.DIR_S if dy > 0 else self.DIR_N

        return direction_map

    def _setup_quadrant_cell_mapping(self):
        """
        Setup mapping from piston quadrants to direction map cells.

        For piston at (i, j):
          - NE quadrant corresponds to cell (i, j) if valid
          - NW quadrant corresponds to cell (i-1, j) if valid
          - SW quadrant corresponds to cell (i-1, j-1) if valid
          - SE quadrant corresponds to cell (i, j-1) if valid
        """
        gx = self.config.GRIDSIZEX
        gy = self.config.GRIDSIZEY
        nx = gx - 1
        ny = gy - 1

        # For each piston and each quadrant, store the cell index or -1 if invalid
        # Shape: (GRIDSIZEX, GRIDSIZEY, 4) where 4 is the quadrant index
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

        Args:
            sensors: Sensor readings [NE, NW, SW, SE] for this piston
            quadrant: Which quadrant we're checking (NE, NW, SW, SE)
            direction: The direction letter for this quadrant

        Returns:
            True if piston should be lowered, False otherwise
        """
        if direction == self.DIR_I:
            return False

        # Get the destination quadrant based on direction
        if direction == self.DIR_S:
            dest_quadrant = self.SOUTH_OF.get(quadrant)
        elif direction == self.DIR_N:
            dest_quadrant = self.NORTH_OF.get(quadrant)
        elif direction == self.DIR_E:
            dest_quadrant = self.EAST_OF.get(quadrant)
        elif direction == self.DIR_W:
            dest_quadrant = self.WEST_OF.get(quadrant)
        else:
            return False

        # If no valid destination (e.g., moving South from SE), skip
        if dest_quadrant is None:
            return False

        # Check conditions: ball in source, no ball in destination
        ball_in_source = sensors[quadrant] > 0
        no_ball_in_dest = sensors[dest_quadrant] == 0

        return ball_in_source and no_ball_in_dest

    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        """
        Compute desired rod height for position (i, j).

        Checks all 4 quadrants and lowers piston if any direction condition is met.
        """
        # Check each quadrant
        for quadrant in [NE, NW, SW, SE]:
            direction = self._get_direction_for_quadrant(i, j, quadrant)
            if self._check_quadrant_action(sensors, quadrant, direction):
                return 0.5  # Lower the piston

        return 1.5  # Raise the piston

    """ def update_all(self, timestep: int, sensors: np.ndarray) -> np.ndarray:
        gx = self.config.GRIDSIZEX
        gy = self.config.GRIDSIZEY

        # Start with neutral height
        desired = np.full((gx, gy), 1.0)

        # Check each quadrant for each piston
        for quadrant in [NE, NW, SW, SE]:
            # Get direction for each piston's quadrant
            cx = self.quadrant_cell_x[:, :, quadrant]
            cy = self.quadrant_cell_y[:, :, quadrant]

            # Valid cells mask
            valid = (cx >= 0) & (cy >= 0)

            # Get directions (use 'I' for invalid)
            directions = np.full((gx, gy), self.DIR_I, dtype='U1')
            directions[valid] = self.direction_map[cy[valid], cx[valid]]

            # Process each direction type
            for direction, dest_map in [
                (self.DIR_S, self.SOUTH_OF),
                (self.DIR_N, self.NORTH_OF),
                (self.DIR_E, self.EAST_OF),
                (self.DIR_W, self.WEST_OF),
            ]:
                dest_quadrant = dest_map.get(quadrant)
                if dest_quadrant is None:
                    continue

                # Mask for this direction
                dir_mask = directions == direction

                if not dir_mask.any():
                    continue

                # Check conditions
                ball_in_source = sensors[:, :, quadrant] > 0
                no_ball_in_dest = sensors[:, :, dest_quadrant] == 0

                # Lower piston where conditions are met
                should_lower = dir_mask & ball_in_source & no_ball_in_dest
                desired[should_lower] = 0.5

        return desired
 """
    def set_direction_map(self, direction_map):
        """Update the direction map at runtime."""
        self.direction_map = np.array(direction_map)

    def get_direction_map(self):
        """Return the current direction map."""
        return self.direction_map.copy()

    def print_direction_map(self):
        """Print the direction map in a readable format."""
        print("Direction Map (Single):")
        print("  " + " ".join(f"{i}" for i in range(self.direction_map.shape[1])))
        # Print from top to bottom (high y to low y)
        for j in range(self.direction_map.shape[0] - 1, -1, -1):
            row = " ".join(self.direction_map[j, :])
            print(f"{j} {row}")
