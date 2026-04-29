"""
Priority Controller

Uses a static priority map and a live running map that persists between timesteps.
The running map remembers the current active direction per cell and updates it
based on real-time sensor feedback, rather than re-evaluating priority from scratch.

Action logic (stateful priority with memory):
  - Running map initialised to first priority direction per cell
  - Each timestep, pistons in the direction of current movement check if the
    path is blocked; if so, the running map switches to the next priority direction
  - Once a higher-priority direction becomes free again, the running map reverts to it
  - Rod is lowered to LOW_HEIGHT when: ball in source AND destination (per running map) is free
  - Otherwise raised to HIGH_HEIGHT

A direction map must always be provided explicitly (see experiments/ for examples).
"""

import numpy as np
from src.controllers.controller_base import Controller
from src.config import NE, NW, SW, SE


class ControllerPriority(Controller):
    """Controller with direction selection based on priority and availability."""

    # Direction constants
    DIR_N = 'N'
    DIR_S = 'S'
    DIR_E = 'E'
    DIR_W = 'W'
    DIR_I = 'I'  # Idle/no action

    # Mapping from quadrant to destination quadrant for each direction
    SOUTH_OF = {NE: SE, NW: SW, SE: None, SW: None}
    NORTH_OF = {SE: NE, SW: NW, NE: None, NW: None}
    EAST_OF = {NW: NE, SW: SE, NE: None, SE: None}
    WEST_OF = {NE: NW, SE: SW, NW: None, SW: None}

    DIRECTION_MAPS = {
        'N': NORTH_OF,
        'S': SOUTH_OF,
        'E': EAST_OF,
        'W': WEST_OF,
    }

    def __init__(self, config, direction_map):
        """
        Initialize the priority direction map controller.

        Args:
            config: SimConfig instance
            direction_map: 2D array of priority direction strings (e.g., 'N', 'NE', 'SEW')
                          Each string is a priority list: first char is preferred.
                          Shape should be (GRIDSIZEY-1, GRIDSIZEX-1) for cells.
        """
        super().__init__(config)
        self.priority_map = np.array(direction_map)

        # Initialize running map with first priority direction for each cell
        self.running_map = np.empty_like(self.priority_map, dtype='U1')
        for j in range(self.priority_map.shape[0]):
            for i in range(self.priority_map.shape[1]):
                priorities = self.priority_map[j, i]
                if priorities and priorities != self.DIR_I:
                    self.running_map[j, i] = priorities[0]
                else:
                    self.running_map[j, i] = self.DIR_I

        # Precompute cell-to-quadrant mapping
        self._setup_quadrant_cell_mapping()

        # Print maps at initialization
        #self.print_direction_map()

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

    def _get_cell_coords(self, i, j, quadrant):
        """Get cell coordinates for a piston's quadrant."""
        cx = self.quadrant_cell_x[i, j, quadrant]
        cy = self.quadrant_cell_y[i, j, quadrant]
        return cx, cy

    def _is_direction_possible(self, sensors, quadrant, direction):
        """
        Check if a direction is possible from a quadrant.
        Returns True if there's a ball in the quadrant and destination is free.
        """
        if direction not in self.DIRECTION_MAPS:
            return False

        dest_map = self.DIRECTION_MAPS[direction]
        dest_quadrant = dest_map.get(quadrant)

        if dest_quadrant is None:
            return False

        # Direction is possible if destination is free
        return sensors[dest_quadrant] == 0

    def _can_piston_update_cell(self, quadrant, current_direction):
        """
        Check if a piston (via this quadrant) can update the running map for a cell.

        Only pistons in the direction of the current movement can update.
        E.g., for direction N, only pistons where the cell is in SE or SW quadrant
        (meaning the piston is to the north of the cell) can update.
        """
        if current_direction not in self.DIRECTION_MAPS:
            return False

        dest_map = self.DIRECTION_MAPS[current_direction]
        # If this quadrant has a valid destination for the current direction,
        # this piston is in the direction of movement and can see if it's blocked
        return dest_map.get(quadrant) is not None

    def _update_running_map_for_cell(self, cx, cy, sensors, quadrant):
        """
        Update the running map for a cell based on current sensor state.

        Only updates if this piston is in the direction of current movement
        (i.e., has visibility to check if path is blocked).

        Always tries to use the highest priority (first) direction that's available.
        Falls back to lower priorities only when higher ones are blocked.
        """
        if cx < 0 or cy < 0:
            return

        priorities = self.priority_map[cy, cx]
        if not priorities or priorities == self.DIR_I:
            return

        current_direction = self.running_map[cy, cx]

        # Only pistons in the direction of movement can update the running map
        if not self._can_piston_update_cell(quadrant, current_direction):
            return

        # Check if there's a ball in this quadrant
        if sensors[quadrant] == 0:
            # No ball here, nothing to update
            return

        # This piston IS in the direction of movement
        # Try each priority direction in order, pick highest priority that's available
        for direction in priorities:
            if direction not in self.DIRECTION_MAPS:
                continue

            dest_map = self.DIRECTION_MAPS[direction]
            dest_quadrant = dest_map.get(quadrant)

            # Check if this direction is valid from this quadrant and destination is free
            if dest_quadrant is not None and sensors[dest_quadrant] == 0:
                self.running_map[cy, cx] = direction
                return

        # No direction possible from this piston's view, keep current

    def _check_quadrant_action(self, sensors, quadrant, direction):
        """Check if current direction triggers a lower action."""
        if direction == self.DIR_I or not direction:
            return False

        if direction not in self.DIRECTION_MAPS:
            return False

        dest_map = self.DIRECTION_MAPS[direction]
        dest_quadrant = dest_map.get(quadrant)

        if dest_quadrant is None:
            return False

        # Check: ball in source, no ball in destination
        ball_in_source = sensors[quadrant] > 0
        no_ball_in_dest = sensors[dest_quadrant] == 0

        return ball_in_source and no_ball_in_dest

    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        """
        Compute desired rod height for position (i, j).
        Updates running map and checks if action should be taken.
        """
        # First, update running map for each quadrant
        for quadrant in [NE, NW, SW, SE]:
            cx, cy = self._get_cell_coords(i, j, quadrant)
            if cx >= 0 and cy >= 0:
                self._update_running_map_for_cell(cx, cy, sensors, quadrant)

        # Then check if any quadrant triggers lowering
        for quadrant in [NE, NW, SW, SE]:
            cx, cy = self._get_cell_coords(i, j, quadrant)
            if cx >= 0 and cy >= 0:
                direction = self.running_map[cy, cx]
                if self._check_quadrant_action(sensors, quadrant, direction):
                    return self.config.LOW_HEIGHT  # Lower the piston

        return self.config.HIGH_HEIGHT  # Raise the piston

    def set_direction_map(self, direction_map):
        """Update the priority map and reset running map."""
        self.priority_map = np.array(direction_map)
        # Reset running map to first priorities
        for j in range(self.priority_map.shape[0]):
            for i in range(self.priority_map.shape[1]):
                priorities = self.priority_map[j, i]
                if priorities and priorities != self.DIR_I:
                    self.running_map[j, i] = priorities[0]
                else:
                    self.running_map[j, i] = self.DIR_I

    def get_direction_map(self):
        """Return the current running map."""
        return self.running_map.copy()

    def get_priority_map(self):
        """Return the static priority map."""
        return self.priority_map.copy()

    def print_direction_map(self):
        """Print both priority and running direction maps."""
        print("Priority Map (Static):")
        max_width = max(len(str(d)) for row in self.priority_map for d in row)
        max_width = max(max_width, 2)
        header = "  " + " ".join(f"{i:>{max_width}}" for i in range(self.priority_map.shape[1]))
        print(header)
        for j in range(self.priority_map.shape[0] - 1, -1, -1):
            row = " ".join(f"{self.priority_map[j, i]:>{max_width}}" for i in range(self.priority_map.shape[1]))
            print(f"{j} {row}")

        print("\nRunning Map (Adaptive):")
        print("  " + " ".join(f"{i}" for i in range(self.running_map.shape[1])))
        for j in range(self.running_map.shape[0] - 1, -1, -1):
            row = " ".join(self.running_map[j, :])
            print(f"{j} {row}")
