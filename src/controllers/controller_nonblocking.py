"""
Non-blocking Controller

Uses a 2D map where each cell contains exactly one direction (N, S, E, W, or I).
Each piston inspects the direction assigned to each of its 4 surrounding quadrants.

Action logic (does not check if destination is free):
  - Ball present in a valid source quadrant for the assigned direction → lower rod to 0.5
  - Otherwise → raise rod to 1.5

Compared to ControllerBlocking, this will push balls even when the
destination cell is already occupied, which can cause crowding but reacts faster.
A direction map must always be provided explicitly (see experiments/ for examples).
"""

import numpy as np
from src.controllers.controller_base import Controller
from src.config import NE, NW, SW, SE


class ControllerNonBlocking(Controller):
    """Controller that uses a direction map without checking destination availability."""

    # Valid source quadrants for each direction
    VALID_SOURCES = {
        'S': {NE, NW},
        'N': {SE, SW},
        'E': {NW, SW},
        'W': {NE, SE},
    }

    def __init__(self, config, direction_map):
        self.direction_map = np.array(direction_map)
        super().__init__(config)
        self.print_direction_map()

    def _check_quadrant_action(self, sensors, quadrant, direction):
        """Return True if ball is in a valid source quadrant for the given direction."""
        valid_sources = self.VALID_SOURCES.get(direction, set())
        return quadrant in valid_sources and sensors[quadrant] > 0

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
        print("Direction Map (NonBlocking):")
        print("  " + " ".join(str(i) for i in range(self.direction_map.shape[1])))
        for j in range(self.direction_map.shape[0] - 1, -1, -1):
            print(f"{j} {' '.join(self.direction_map[j, :])}")
