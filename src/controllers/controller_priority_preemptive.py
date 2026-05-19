"""
Priority-Preemptive Controller

Combines the per-cell direction priority lists from the priority controller with
a global rule-level direction priority used to break symmetric pass-by deadlocks
(e.g. a W-mover and an E-mover bracketing the same rod).

Two layers of priority:
  - Per-cell (priority_map): each cell carries an ordered list of directions
    (e.g. "NE" = prefer N, fall back to E if N is blocked). A running map tracks
    the currently active direction per cell and updates whenever the preferred
    destination is/becomes blocked.
  - Rule-level (DIRECTION_PRIORITY): a global ranking over directions
    (E > S > W > N) used at each rod to suppress lower-priority rules in favor
    of higher-priority neighbors.

Each rod update:
  1. Refresh the running map for each surrounding quadrant based on current
     sensor state.
  2. Find Dmax = the highest-priority direction (per DIRECTION_PRIORITY) among
     the running directions of quadrants where a ball is present *and* the ball's
     rule actually fires at this rod (destination quadrant is defined for this
     rod and is free). Balls that cannot use this rod — whether because the
     destination is occupied by another ball or because the rod has no rule for
     the ball's quadrant×direction — are excluded from the priority claim, so
     they cannot stall lower-priority neighbors when they are themselves stuck.
  3. Evaluate the standard rule only for balls whose running direction is Dmax.
     Lower the rod if any such rule fires; otherwise stay raised. Quadrants
     whose ball moves in a lower-priority direction are ignored even if their
     own rule would have fired.

Single-char direction strings still work — they simply have no fallback.

A priority map must always be provided explicitly (see experiments/ for examples).
"""

import numpy as np
from src.controllers.controller_base import Controller
from src.config import NE, NW, SW, SE


class ControllerPriorityPreemptive(Controller):
    """Direction-map controller with per-cell fallback priority and global rule-level priority."""

    DIR_I = 'I'  # Idle / no action

    # Destination quadrant for each source quadrant per direction
    SOUTH_OF = {NE: SE, NW: SW, SE: None, SW: None}
    NORTH_OF = {SE: NE, SW: NW, NE: None, NW: None}
    EAST_OF  = {NW: NE, SW: SE, NE: None, SE: None}
    WEST_OF  = {NE: NW, SE: SW, NW: None, SW: None}

    DIRECTION_MAPS = {
        'N': NORTH_OF,
        'S': SOUTH_OF,
        'E': EAST_OF,
        'W': WEST_OF,
    }

    # Rule-level direction priority — earlier entries preempt later ones.
    DIRECTION_PRIORITY = ['E', 'S', 'W', 'N']

    def __init__(self, config, direction_map):
        super().__init__(config)
        self.priority_map = np.array(direction_map)

        # Running map: currently active single-char direction per cell.
        self.running_map = np.empty_like(self.priority_map, dtype='U1')
        for j in range(self.priority_map.shape[0]):
            for i in range(self.priority_map.shape[1]):
                priorities = self.priority_map[j, i]
                if priorities and priorities != self.DIR_I:
                    self.running_map[j, i] = priorities[0]
                else:
                    self.running_map[j, i] = self.DIR_I

        self.print_direction_map()

    def _get_cell_coords(self, i, j, quadrant):
        cx = self.quadrant_cell_x[i, j, quadrant]
        cy = self.quadrant_cell_y[i, j, quadrant]
        return cx, cy

    def _can_piston_update_cell(self, quadrant, current_direction):
        """Only pistons that lie in the direction of current motion can update the
        running map — they're the ones whose sensors can see whether the path is blocked."""
        if current_direction not in self.DIRECTION_MAPS:
            return False
        dest_map = self.DIRECTION_MAPS[current_direction]
        return dest_map.get(quadrant) is not None

    def _update_running_map_for_cell(self, cx, cy, sensors, quadrant):
        """Pick the highest cell-priority direction whose destination is free from this piston's view."""
        if cx < 0 or cy < 0:
            return

        priorities = self.priority_map[cy, cx]
        if not priorities or priorities == self.DIR_I:
            return

        current_direction = self.running_map[cy, cx]
        if not self._can_piston_update_cell(quadrant, current_direction):
            return

        if sensors[quadrant] == 0:
            return

        for direction in priorities:
            if direction not in self.DIRECTION_MAPS:
                continue
            dest_map = self.DIRECTION_MAPS[direction]
            dest_quadrant = dest_map.get(quadrant)
            if dest_quadrant is not None and sensors[dest_quadrant] == 0:
                self.running_map[cy, cx] = direction
                return

    def _rule_fires(self, sensors, quadrant, direction):
        """True if ball in source quadrant AND destination quadrant is free."""
        if direction == self.DIR_I or direction not in self.DIRECTION_MAPS:
            return False
        dest_map = self.DIRECTION_MAPS[direction]
        dest_quadrant = dest_map.get(quadrant)
        if dest_quadrant is None:
            return False
        return sensors[quadrant] > 0 and sensors[dest_quadrant] == 0

    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        # Step 1: refresh running map for each surrounding quadrant.
        for quadrant in [NE, NW, SW, SE]:
            cx, cy = self._get_cell_coords(i, j, quadrant)
            if cx >= 0 and cy >= 0:
                self._update_running_map_for_cell(cx, cy, sensors, quadrant)

        # Step 2: collect (quadrant, running_direction) pairs whose rule actually
        # fires at this rod (destination defined and free). Balls that can't use
        # this rod — either no rule from this quadrant×direction, or rule blocked
        # by another ball — don't claim priority. This avoids stalling lower-priority
        # neighbors behind a ball that is locally stuck.
        present = []
        for quadrant in [NE, NW, SW, SE]:
            if sensors[quadrant] == 0:
                continue
            cx, cy = self._get_cell_coords(i, j, quadrant)
            if cx < 0 or cy < 0:
                continue
            direction = self.running_map[cy, cx]
            if direction not in self.DIRECTION_PRIORITY:
                continue
            dest_quadrant = self.DIRECTION_MAPS[direction].get(quadrant)
            if dest_quadrant is None or sensors[dest_quadrant] > 0:
                continue
            present.append((quadrant, direction))

        if not present:
            return self.config.HIGH_HEIGHT

        # Step 3: pick Dmax and evaluate rules only for that direction.
        dmax = min(present, key=lambda qd: self.DIRECTION_PRIORITY.index(qd[1]))[1]
        for quadrant, direction in present:
            if direction == dmax and self._rule_fires(sensors, quadrant, direction):
                return self.config.LOW_HEIGHT
        return self.config.HIGH_HEIGHT

    def set_direction_map(self, direction_map):
        self.priority_map = np.array(direction_map)
        for j in range(self.priority_map.shape[0]):
            for i in range(self.priority_map.shape[1]):
                priorities = self.priority_map[j, i]
                if priorities and priorities != self.DIR_I:
                    self.running_map[j, i] = priorities[0]
                else:
                    self.running_map[j, i] = self.DIR_I

    def get_direction_map(self):
        return self.running_map.copy()

    def get_priority_map(self):
        return self.priority_map.copy()

    def print_direction_map(self):
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
