"""
Greedy Bin Phased Controller

Single-lane greedy bin-covering with a two-phase, mass-aware gate.

Like ControllerGreedyBin, top-row cells are 'B' (active bin) or 'I' (latched
closed); mid-row gate cells south of each bin flip between 'N' (gate open,
ball climbs into bin) and 'E' (gate closed, ball passes east). Latching is
monotonic: a bin transitions 'B' -> 'I' only when its accumulated weight has
reached TARGET_WEIGHT, and never reopens.

What differs from the baseline is the per-tick gate decision. Each active bin
reads its own accumulated weight and the mass of the ball currently waiting in
the mid cell directly below:

  bin_weight    = sensors[col, top_row, NE]
  incoming_mass = sensors[col, top_row, SE]

The gate opens this tick iff one of two phase conditions holds:

  Phase 1 (fill-any):    bin_weight + incoming_mass <= TARGET_WEIGHT
                         -> still safely under target, take anything

  Phase 2 (mop-up):      incoming_mass <= OVERSHOOT_TOLERANCE
                         -> overshoot is inevitable but the increment is small
                            enough to accept and latch closed

Otherwise the gate closes for this tick and the ball continues east through
the travel lane, around the W return, and gets another chance at this bin or
a later one. The phases are implicit in the disjunction — no explicit phase
variable is tracked.

Locality: each bin's decision reads only its own NE/SE sensor quadrants and
the constants TARGET_WEIGHT / OVERSHOOT_TOLERANCE. No cross-bin coordination.
"""

import numpy as np
from src.controllers.controller_blocking import ControllerBlocking
from src.config import NE, NW, SW, SE


class ControllerGreedyBinPhased(ControllerBlocking):

    def __init__(self, config, direction_map):
        super().__init__(config, direction_map)
        self._effective_map = self.direction_map.copy()
        n_rows = self.direction_map.shape[0]
        self._top_row = n_rows - 1
        self._mid_row = n_rows - 2
        # Static defaults for the middle row at non-gate columns; gate columns
        # are driven per-tick by _update_gates.
        self._middle_default = self.direction_map[self._mid_row, :].copy()
        for col in range(self.direction_map.shape[1]):
            if self.direction_map[self._top_row, col] == 'B':
                self._middle_default[col] = 'E'
        self._bin_cols = [c for c in range(self.direction_map.shape[1])
                          if self.direction_map[self._top_row, c] == 'B']
        # Start every gate open so the first arrival is admitted into the fill phase.
        for col in self._bin_cols:
            self._effective_map[self._mid_row, col] = 'N'
        print(f"Greedy bin phased controller: target={self.config.TARGET_WEIGHT} kg, "
              f"overshoot_tolerance={self.config.OVERSHOOT_TOLERANCE} kg, "
              f"{len(self._bin_cols)} bins at cols {self._bin_cols}")

    def _update_gates(self, sensors):
        target = self.config.TARGET_WEIGHT
        tolerance = self.config.OVERSHOOT_TOLERANCE
        for col in self._bin_cols:
            if self._effective_map[self._top_row, col] != 'B':
                continue  # already latched 'I'; stays closed forever

            bin_weight = float(sensors[col, self._top_row, NE])
            if bin_weight >= target:
                # Cleared the lower-bound. Latch closed and restore the static
                # E-pass direction so future arrivals just flow past.
                self._effective_map[self._top_row, col] = 'I'
                self._effective_map[self._mid_row, col] = self._middle_default[col]
                continue

            incoming_mass = float(sensors[col, self._top_row, SE])
            if incoming_mass == 0.0:
                # Nothing waiting -> keep gate open so the next arrival is taken.
                self._effective_map[self._mid_row, col] = 'N'
                continue

            safe_accept  = (bin_weight + incoming_mass) <= target
            mopup_accept = incoming_mass <= tolerance
            if safe_accept or mopup_accept:
                self._effective_map[self._mid_row, col] = 'N'
            else:
                # Would overshoot AND the ball is too heavy to accept as closure.
                # Reject: send it east to recirculate.
                self._effective_map[self._mid_row, col] = 'E'

    def _get_direction_for_quadrant(self, i, j, quadrant):
        cx = self.quadrant_cell_x[i, j, quadrant]
        cy = self.quadrant_cell_y[i, j, quadrant]
        if cx < 0 or cy < 0:
            return 'I'
        return self._effective_map[cy, cx]

    def update(self, i, j, timestep, sensors):
        # Same blocking firing rule as ControllerGreedyBin: a destination cell
        # currently in state 'B' counts as "free" so an active bin can keep
        # accepting balls as it fills.
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
        # Northmost piston row: low at piston columns adjacent to a bin whose
        # gate is actually firing this tick (bin still 'B', gate open, ball
        # present in the mid cell). When the gate is 'E' (rejecting), keep the
        # north piston high so the bin stays walled.
        n_cells = self.direction_map.shape[1]
        is_active  = self._effective_map[self._top_row, :] == 'B'
        gate_open  = self._effective_map[self._mid_row, :] == 'N'
        ball_in_mid = sensors[:n_cells, self._top_row, SE] > 0
        n_flow = is_active & gate_open & ball_in_mid
        left_nflow  = np.concatenate([[False], n_flow])
        right_nflow = np.concatenate([n_flow, [False]])
        north_low = left_nflow | right_nflow
        desired[:, -1] = np.where(north_low, self.config.LOW_HEIGHT, self.config.HIGH_HEIGHT)
        return desired
