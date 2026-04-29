"""
Non-blocking Controller

Uses a 2D map where each cell contains exactly one direction (N, S, E, W, or I).
Each piston inspects the direction assigned to each of its 4 surrounding quadrants.

Action logic (does not check if destination is free):
  - Ball present in a valid source quadrant for the assigned direction → lower rod to LOW_HEIGHT
  - Otherwise → raise rod to HIGH_HEIGHT

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
        self._precompute_vectorized_tables()
        self.print_direction_map()

    def _check_quadrant_action(self, sensors, quadrant, direction):
        """Return True if ball is in a valid source quadrant for the given direction."""
        valid_sources = self.VALID_SOURCES.get(direction, set())
        return quadrant in valid_sources and sensors[quadrant] > 0

    def _precompute_vectorized_tables(self):
        """Build per-quadrant boolean masks for vectorized update_all()."""
        gx = self.config.GRIDSIZEX
        gy = self.config.GRIDSIZEY
        # For each direction, build a (gx, gy) bool mask that is True
        # when ANY valid-source quadrant for that direction has a matching
        # direction in the map.  At runtime we just AND with sensor > 0.
        self._quadrant_dir_match = {}  # direction -> (gx, gy, 4) bool
        for direction, sources in self.VALID_SOURCES.items():
            match = np.zeros((gx, gy, 4), dtype=bool)
            for q in range(4):
                if q not in sources:
                    continue
                cx = self.quadrant_cell_x[:, :, q]
                cy = self.quadrant_cell_y[:, :, q]
                valid = (cx >= 0) & (cy >= 0)
                # For valid cells, check if direction_map matches
                cx_safe = np.clip(cx, 0, self.direction_map.shape[1] - 1)
                cy_safe = np.clip(cy, 0, self.direction_map.shape[0] - 1)
                match[:, :, q] = valid & (self.direction_map[cy_safe, cx_safe] == direction)
            self._quadrant_dir_match[direction] = match

    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        for quadrant in [NE, NW, SW, SE]:
            direction = self._get_direction_for_quadrant(i, j, quadrant)
            if self._check_quadrant_action(sensors, quadrant, direction):
                return self.config.LOW_HEIGHT
        return self.config.HIGH_HEIGHT

    def update_all(self, timestep, sensors):
        """Vectorized update for all rods at once."""
        has_ball = sensors > 0  # (gx, gy, 4)
        lower = np.zeros((self.config.GRIDSIZEX, self.config.GRIDSIZEY), dtype=bool)
        for direction, match in self._quadrant_dir_match.items():
            # match is (gx, gy, 4) bool — True where quadrant q is a valid source
            # and its direction_map cell == direction.
            # OR over quadrants: any quadrant triggers → lower
            lower |= np.any(match & has_ball, axis=2)
        return np.where(lower, self.config.LOW_HEIGHT, self.config.HIGH_HEIGHT)

    def set_direction_map(self, direction_map):
        self.direction_map = np.array(direction_map)

    def get_direction_map(self):
        return self.direction_map.copy()

    def print_direction_map(self):
        print("Direction Map (NonBlocking):")
        print("  " + " ".join(str(i) for i in range(self.direction_map.shape[1])))
        for j in range(self.direction_map.shape[0] - 1, -1, -1):
            print(f"{j} {' '.join(self.direction_map[j, :])}")
