"""
Blocking Controller

Uses a 2D map where each cell contains exactly one direction (N, S, E, W, or I).
Each piston inspects the direction assigned to each of its 4 surrounding quadrants.

Action logic (avoids pushing into occupied cells):
  - Ball present in source quadrant AND destination quadrant is free → lower rod to 0.5
  - Otherwise → raise rod to 1.5

A direction map must always be provided explicitly (see experiments/ for examples).
"""

import numpy as np
from src.controllers.controller_base import Controller
from src.config import NE, NW, SW, SE


class ControllerBlocking(Controller):
    """Controller that uses a direction map to guide ball movement (checks destination is free)."""

    # Destination quadrant for each source quadrant per direction
    SOUTH_OF = {NE: SE, NW: SW, SE: None, SW: None}
    NORTH_OF = {SE: NE, SW: NW, NE: None, NW: None}
    EAST_OF  = {NW: NE, SW: SE, NE: None, SE: None}
    WEST_OF  = {NE: NW, SE: SW, NW: None, SW: None}

    def __init__(self, config, direction_map):
        self.direction_map = np.array(direction_map)
        super().__init__(config)
        self.print_direction_map()

    def _check_quadrant_action(self, sensors, quadrant, direction):
        """Return True if piston should lower: ball in source and destination is free."""
        dest_map = {'S': self.SOUTH_OF, 'N': self.NORTH_OF,
                    'E': self.EAST_OF,  'W': self.WEST_OF}.get(direction)
        if dest_map is None:
            return False
        dest_quadrant = dest_map.get(quadrant)
        if dest_quadrant is None:
            return False
        return sensors[quadrant] > 0 and sensors[dest_quadrant] == 0

    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        for quadrant in [NE, NW, SW, SE]:
            direction = self._get_direction_for_quadrant(i, j, quadrant)
            if self._check_quadrant_action(sensors, quadrant, direction):
                return 0.5
        return 1.5

    def set_direction_map(self, direction_map):
        self.direction_map = np.array(direction_map)

    def get_direction_map(self):
        return self.direction_map.copy()

    def print_direction_map(self):
        print("Direction Map (Blocking):")
        print("  " + " ".join(str(i) for i in range(self.direction_map.shape[1])))
        for j in range(self.direction_map.shape[0] - 1, -1, -1):
            print(f"{j} {' '.join(self.direction_map[j, :])}")
