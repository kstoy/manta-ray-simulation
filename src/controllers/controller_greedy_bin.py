"""
Greedy Bin Controller

Local first-fit bin-covering baseline.

Top-row cells are either 'B' (active bin, still filling) or 'I' (parked,
either a separator or a bin that has reached its target). When a bin
crosses TARGET_WEIGHT, its top cell latches from 'B' to 'I' and never
reopens — closing is monotonic.

Mid-row gate cells directly south of a bin column track that bin's state:
'N' (gate open) while the bin is 'B', and 'E' (gate closed) once it is 'I'.

Firing rule: blocking everywhere — a piston only lowers if the destination
quadrant is free — with one exception. A destination cell in state 'B'
counts as "free" for the purpose of this check, so an active bin can keep
accepting balls as it fills. Every other transition (E in the travel lane,
W in the return lane, N/S at the corner climbs/drops, and an 'I'-state
closed bin) is strictly blocking, which is what prevents pushing balls
into occupied cells along the recirculation loop.

The northmost piston row goes low at piston columns where an active 'B'
bin's gate is firing (gate open AND ball present in the mid cell directly
south) — pulling the entering ball against the north wall. Otherwise the
north piston stays HIGH and walls in whatever is already collected.

Locality: each bin's state transition reads only its own weight; the
gate sync reads only that bin's state; the north-piston rule reads only
the two adjacent top-row cells.
"""

import numpy as np
from src.controllers.controller_blocking import ControllerBlocking
from src.config import NE, NW, SW, SE


class ControllerGreedyBin(ControllerBlocking):

    def __init__(self, config, direction_map):
        super().__init__(config, direction_map)
        self._effective_map = self.direction_map.copy()
        # Top row holds the bins; mid row holds the gates directly below them.
        # Any rows further south (return lanes, buffers) are not addressed by name here.
        n_rows = self.direction_map.shape[0]
        self._top_row = n_rows - 1
        self._mid_row = n_rows - 2
        # Static defaults for the middle row (E for separators / entry, S at east drop).
        # Gate columns will be overwritten by the bin-state-driven update each tick.
        self._middle_default = self.direction_map[self._mid_row, :].copy()
        for col in range(self.direction_map.shape[1]):
            if self.direction_map[self._top_row, col] == 'B':
                self._middle_default[col] = 'E'
        # Bin columns: where the initial top row has 'B'.
        self._bin_cols = [c for c in range(self.direction_map.shape[1])
                          if self.direction_map[self._top_row, c] == 'B']
        # Sync gate cells to their initial bin state ('B' → 'N').
        for col in self._bin_cols:
            self._effective_map[self._mid_row, col] = 'N'
        print(f"Greedy bin controller: target={self.config.TARGET_WEIGHT} kg, "
              f"{len(self._bin_cols)} bins at cols {self._bin_cols}")

    def _update_gates(self, sensors):
        target = self.config.TARGET_WEIGHT
        for col in self._bin_cols:
            if self._effective_map[self._top_row, col] != 'B':
                continue  # already latched 'I'; stays closed
            bin_weight = float(sensors[col, self._top_row, NE])
            if bin_weight >= target:
                self._effective_map[self._top_row, col] = 'I'
                self._effective_map[self._mid_row, col] = self._middle_default[col]

    def _get_direction_for_quadrant(self, i, j, quadrant):
        cx = self.quadrant_cell_x[i, j, quadrant]
        cy = self.quadrant_cell_y[i, j, quadrant]
        if cx < 0 or cy < 0:
            return 'I'
        return self._effective_map[cy, cx]

    def update(self, i, j, timestep, sensors):
        # Blocking firing rule with the 'B'-bin exception: a destination cell
        # currently in state 'B' accepts a ball regardless of its occupancy,
        # so balls can stack inside an active bin. Every other transition
        # requires the destination quadrant to be empty.
        for quadrant in [NE, NW, SW, SE]:
            direction = self._get_direction_for_quadrant(i, j, quadrant)
            dest_map = {'S': self.SOUTH_OF, 'N': self.NORTH_OF,
                        'E': self.EAST_OF,  'W': self.WEST_OF}.get(direction)
            if dest_map is None:
                continue
            dest_quadrant = dest_map.get(quadrant)
            if dest_quadrant is None:
                continue
            if sensors[quadrant] == 0:
                continue
            dx = self.quadrant_cell_x[i, j, dest_quadrant]
            dy = self.quadrant_cell_y[i, j, dest_quadrant]
            dest_is_active_bin = (dx >= 0 and dy >= 0
                                  and self._effective_map[dy, dx] == 'B')
            if dest_is_active_bin or sensors[dest_quadrant] == 0:
                return self.config.LOW_HEIGHT
        return self.config.HIGH_HEIGHT

    def update_all(self, timestep, sensors):
        self._update_gates(sensors)
        desired = np.empty((self.config.GRIDSIZEX, self.config.GRIDSIZEY))
        for i in range(self.config.GRIDSIZEX):
            for j in range(self.config.GRIDSIZEY):
                desired[i, j] = self.update(i, j, timestep, sensors[i, j])
        # Northmost piston row goes low only at piston columns adjacent to a
        # 'B' bin whose gate is actively firing — gate open AND ball currently
        # in the mid cell directly south of that bin. E-flow lowering from an
        # upstream separator does NOT trigger the follow.
        n_cells = self.direction_map.shape[1]
        is_active = self._effective_map[self._top_row, :] == 'B'
        ball_in_mid = sensors[:n_cells, self._top_row, SE] > 0
        n_flow = is_active & ball_in_mid
        left_nflow  = np.concatenate([[False], n_flow])
        right_nflow = np.concatenate([n_flow, [False]])
        north_low = left_nflow | right_nflow
        desired[:, -1] = np.where(north_low, self.config.LOW_HEIGHT, self.config.HIGH_HEIGHT)
        return desired
