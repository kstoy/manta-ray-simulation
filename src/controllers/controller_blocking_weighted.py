"""
Blocking Controller with weight-based sorter cells.

Extension of ControllerBlocking: most cells behave exactly like the blocking
controller, but a set of cells flagged as "sorters" pick their direction at
runtime from the total mass currently resting on the cell.

A sorter is configured by:
    thresholds: ascending list of mass cutoffs [t1, t2, ...]
    directions: list of direction chars, len = len(thresholds) + 1

Selection: the first threshold that the measured weight is strictly less than
picks the corresponding direction; weights >= the last threshold get the last
direction.

The chosen direction LATCHES on first contact (first timestep the cell sees
non-zero weight) and is held until the cell empties, so the routing is stable
even though the per-frame weight fluctuates while the ball rolls in/out.
"""

import numpy as np
from src.controllers.controller_blocking import ControllerBlocking
from src.config import NE


class ControllerBlockingWeighted(ControllerBlocking):

    def __init__(self, config, direction_map, sorter_map=None):
        self.sorter_map = dict(sorter_map) if sorter_map else {}
        super().__init__(config, direction_map)
        # Mutable copy used by piston lookups; sorter cells get overwritten in-place.
        self._effective_map = self.direction_map.copy()
        self._latched = {}
        if self.sorter_map:
            print(f"Sorter cells: {sorted(self.sorter_map.keys())}")

    def _update_sorters(self, sensors):
        """Latch or release each sorter's direction based on current cell weight."""
        for (cx, cy), cfg in self.sorter_map.items():
            # Cell weight = mass of balls whose center falls in (cx, cy).
            # The NE quadrant of the piston at (cx, cy) accumulates exactly that.
            weight = float(sensors[cx, cy, NE])

            if weight <= 0.0:
                if (cx, cy) in self._latched:
                    del self._latched[(cx, cy)]
                self._effective_map[cy, cx] = self.direction_map[cy, cx]
                continue

            if (cx, cy) in self._latched:
                continue

            direction = self._pick_direction(cfg, weight)
            self._latched[(cx, cy)] = direction
            self._effective_map[cy, cx] = direction

    @staticmethod
    def _pick_direction(sorter_cfg, weight):
        thresholds = sorter_cfg['thresholds']
        directions = sorter_cfg['directions']
        assert len(directions) == len(thresholds) + 1, \
            "sorter_map: len(directions) must be len(thresholds) + 1"
        for t, d in zip(thresholds, directions):
            if weight < t:
                return d
        return directions[-1]

    def _get_direction_for_quadrant(self, i, j, quadrant):
        cx = self.quadrant_cell_x[i, j, quadrant]
        cy = self.quadrant_cell_y[i, j, quadrant]
        if cx < 0 or cy < 0:
            return 'I'
        return self._effective_map[cy, cx]

    def update_all(self, timestep, sensors):
        self._update_sorters(sensors)
        return super().update_all(timestep, sensors)
