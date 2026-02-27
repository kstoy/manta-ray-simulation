"""
Priority Delayed Controller

Identical to ControllerPriority, but a piston will only lower after the action
condition (ball in source, destination free) has been continuously true for at
least `delay_seconds`.  This prevents the surface from reacting instantly to a
passing gap and introduces a deliberate lag that creates more chaotic ball flow.
"""

import numpy as np
from src.controllers.controller_priority import ControllerPriority
from src.config import NE, NW, SW, SE


class ControllerPriorityDelayed(ControllerPriority):
    """Priority controller with a per-piston-quadrant action delay."""

    def __init__(self, config, direction_map, delay_seconds=1.0):
        super().__init__(config, direction_map)
        self.delay_seconds = delay_seconds
        # Timestep when the action condition first became true for each (i, j, quadrant).
        # -1 means the condition was not true on the previous call.
        self._action_since = np.full(
            (config.GRIDSIZEX, config.GRIDSIZEY, 4), -1, dtype=int
        )

    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        # Update the running direction map (same as parent).
        for quadrant in [NE, NW, SW, SE]:
            cx, cy = self._get_cell_coords(i, j, quadrant)
            if cx >= 0 and cy >= 0:
                self._update_running_map_for_cell(cx, cy, sensors, quadrant)

        # Check each quadrant; lower only after the delay has elapsed.
        delay_steps = self.delay_seconds / self.config.DT
        for quadrant in [NE, NW, SW, SE]:
            cx, cy = self._get_cell_coords(i, j, quadrant)
            if cx < 0 or cy < 0:
                continue
            direction = self.running_map[cy, cx]
            if self._check_quadrant_action(sensors, quadrant, direction):
                if self._action_since[i, j, quadrant] == -1:
                    self._action_since[i, j, quadrant] = timestep
                if (timestep - self._action_since[i, j, quadrant]) >= delay_steps:
                    return 0.5
            else:
                self._action_since[i, j, quadrant] = -1

        return 1.5
