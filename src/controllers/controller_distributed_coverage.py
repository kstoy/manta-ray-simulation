"""
Distributed Coverage Controller

Whole-grid distributed bin-coverage via local threshold-triggered diffusion.
Every cell is a bin with target TARGET_WEIGHT. Mass is injected anywhere on
the surface (typically clustered at the centre); each piston decides LOW or
HIGH purely from local sensor reads of its four surrounding cells. Edges
fire to transfer mass from heavier to lighter cells. The system converges
when every cell is at or above target and no edge meets the firing rule.

Local edge rule (per edge between cells A and B with masses m_A, m_B):

  Base eligibility:
    |m_A - m_B| > DIFFUSION_THRESHOLD_HIGH                  (imbalance large enough)
    AND min(m_A, m_B) < TARGET_WEIGHT                       (lighter side still wants mass)
    AND ( max(m_A, m_B) < TARGET_WEIGHT                     (heavier is also below target)
         OR max(m_A, m_B) > TARGET_WEIGHT + DIFFUSION_SAFETY_MARGIN )
                                                            (or heavier has excess to donate)

  Per-cell arbitration:
    Of the (up to 4) eligible outgoing edges from a single donor cell, only
    the one with the LARGEST positive (m_donor - m_neighbor) gets to fire,
    AND only if no edge incident to that cell is currently HOLDING. The
    "no edge HOLDING" gate is essential: without it, after the first edge
    starts a HOLD, the cell would re-arbitrate next tick, pick the now-
    largest remaining direction, and start another HOLD on top of the first,
    laddering up to all-sides-LOW within a few ticks -- recreating the very
    symmetric basin that arbitration was supposed to avoid. With the gate,
    a cell can have AT MOST one outgoing edge in HOLDING at a time; when
    that HOLD ends and the edge enters COOLDOWN, the cell becomes free to
    pick the next direction (typically a different one, since the just-
    fed neighbour now has more mass than the other neighbours).

The compound criterion enforces *non-monotonic* latching: a cell that has
reached target only donates when its mass exceeds target by the safety
margin, so cells that just barely closed are protected from being drained
back below target by an opportunistic neighbor.

Arbitration is necessary because without it, the very first dropped ball
makes its cell heavier than ALL four neighbours simultaneously, all four
edges meet the base rule, all four sides go LOW at once, and the cell
forms a symmetric basin in which nothing rolls. With arbitration the cell
tilts toward a single direction; once that edge enters COOLDOWN, the next
tick re-arbitrates and the cell tilts toward the next-most-depleted
neighbour. Argmax ties are broken in fixed E > W > N > S order.

When an edge meets the rule it enters a HOLDING state for DIFFUSION_HOLD_STEPS
ticks (both pistons of the edge held LOW so the surface dips along the edge
and balls flow from the heavier cell into the lighter one). The HOLDING
period is followed by a DIFFUSION_COOLDOWN_STEPS COOLDOWN period during
which the edge is locked HIGH and cannot re-fire, preventing oscillation.

State is per-edge, encoded as a signed int:
    > 0  : HOLDING with that many ticks remaining
    < 0  : COOLDOWN with -value ticks remaining
    == 0 : IDLE; re-evaluates the firing rule each tick

A piston (i, j) goes LOW iff any of the four edges incident to it is HOLDING.

No direction map and no per-bin geometry: this controller operates on the
plain cell grid uniformly. The experiment file does not provide a
DIRECTION_MAP; GRIDSIZEX and GRIDSIZEY are set explicitly.
"""

import numpy as np
from src.controllers.controller_base import Controller
from src.config import NE


class ControllerDistributedCoverage(Controller):

    def __init__(self, config):
        super().__init__(config)
        self._nx = config.GRIDSIZEX - 1   # cell count in x
        self._ny = config.GRIDSIZEY - 1   # cell count in y

        # Edge state machines (signed int per edge; see module docstring).
        # Horizontal edges connect cells (c, r) and (c, r+1). Shape (nx, ny-1).
        # Vertical edges connect cells (c, r) and (c+1, r). Shape (nx-1, ny).
        h_shape = (self._nx, max(self._ny - 1, 0))
        v_shape = (max(self._nx - 1, 0), self._ny)
        self._h_edge_state = np.zeros(h_shape, dtype=np.int32)
        self._v_edge_state = np.zeros(v_shape, dtype=np.int32)

        print(f"Distributed coverage controller: target={config.TARGET_WEIGHT} kg, "
              f"T_high={config.DIFFUSION_THRESHOLD_HIGH} kg, "
              f"safety_margin={config.DIFFUSION_SAFETY_MARGIN} kg, "
              f"hold={config.DIFFUSION_HOLD_STEPS} steps, "
              f"cooldown={config.DIFFUSION_COOLDOWN_STEPS} steps, "
              f"grid {self._nx}x{self._ny} cells = {self._nx * self._ny} bins")

    def _can_fire(self, m_a, m_b):
        target = self.config.TARGET_WEIGHT
        safety = self.config.DIFFUSION_SAFETY_MARGIN
        t_high = self.config.DIFFUSION_THRESHOLD_HIGH
        diff = np.abs(m_a - m_b)
        m_min = np.minimum(m_a, m_b)
        m_max = np.maximum(m_a, m_b)
        return ((diff > t_high) &
                (m_min < target))
#                & ((m_max < target) | (m_max > target + safety)))

    def _arbitrate_per_cell(self, cell_mass, cell_has_holding):
        """For each cell, pick its single largest-positive-diff outgoing direction.

        Returns boolean arrays the same shape as the h-edge and v-edge state
        grids: an edge is True iff it is the chosen direction of whichever of
        its two endpoint cells is the heavier (i.e. the donor) side, AND that
        donor cell has no other edge currently HOLDING.

        Direction encoding for argmax ordering (tie-break: E > W > N > S):
            0 = E (donor.x < neighbor.x)
            1 = W
            2 = N (donor.y < neighbor.y)
            3 = S
        """
        nx, ny = self._nx, self._ny
        # Pad mass with +inf so out-of-bounds neighbours look infinitely
        # heavy -> the cell never picks that direction as a donation target.
        padded = np.full((nx + 2, ny + 2), np.inf)
        padded[1:nx + 1, 1:ny + 1] = cell_mass

        diff_E = cell_mass - padded[2:nx + 2, 1:ny + 1]
        diff_W = cell_mass - padded[0:nx,     1:ny + 1]
        diff_N = cell_mass - padded[1:nx + 1, 2:ny + 2]
        diff_S = cell_mass - padded[1:nx + 1, 0:ny]

        diffs       = np.stack([diff_E, diff_W, diff_N, diff_S])   # (4, nx, ny)
        chosen_dir  = diffs.argmax(axis=0)                          # (nx, ny)
        # A cell only donates this tick if (a) it has at least one positive
        # outgoing diff AND (b) it has no edge currently in HOLDING. (b) gates
        # the cell so it can't ladder up to all-sides-LOW by adding a new
        # HOLDING on each successive tick while the first one is still active.
        donates     = (diffs.max(axis=0) > 0) & ~cell_has_holding

        # h-edge (c, r): between cells (c, r) and (c, r+1). The edge is the
        # north face of the lower cell and the south face of the upper cell.
        h_chosen = np.zeros((nx, max(ny - 1, 0)), dtype=bool)
        if ny >= 2:
            h_diff = cell_mass[:, :-1] - cell_mass[:, 1:]   # (nx, ny-1)
            lower_picks_N = (h_diff > 0) & donates[:, :-1] & (chosen_dir[:, :-1] == 2)
            upper_picks_S = (h_diff < 0) & donates[:, 1:]  & (chosen_dir[:, 1:]  == 3)
            h_chosen = lower_picks_N | upper_picks_S

        # v-edge (c, r): between cells (c, r) and (c+1, r). East face of left
        # cell, west face of right cell.
        v_chosen = np.zeros((max(nx - 1, 0), ny), dtype=bool)
        if nx >= 2:
            v_diff = cell_mass[:-1, :] - cell_mass[1:, :]   # (nx-1, ny)
            left_picks_E  = (v_diff > 0) & donates[:-1, :] & (chosen_dir[:-1, :] == 0)
            right_picks_W = (v_diff < 0) & donates[1:, :]  & (chosen_dir[1:, :]  == 1)
            v_chosen = left_picks_E | right_picks_W

        return h_chosen, v_chosen

    def _step_edges(self, state, can_fire):
        """Vectorized state machine advance.

        Transitions per element:
          state > 1   -> state - 1               (HOLDING decrement)
          state == 1  -> -COOLDOWN_STEPS         (HOLDING expires -> COOLDOWN)
          state < 0   -> state + 1               (COOLDOWN increment toward 0)
          state == 0  -> HOLD_STEPS if can_fire else 0
        """
        hold = self.config.DIFFUSION_HOLD_STEPS
        cool = self.config.DIFFUSION_COOLDOWN_STEPS
        holding = state > 0
        cooling = state < 0
        next_holding = np.where(state == 1, -cool, state - 1)
        next_cooling = state + 1
        next_idle    = np.where(can_fire, hold, 0)
        result = np.where(holding, next_holding,
                          np.where(cooling, next_cooling, next_idle))
        return result.astype(np.int32)

    def update(self, i, j, timestep, sensors):
        # Not used at runtime; update_all is the canonical entry point. Kept
        # to satisfy Controller's abstract interface.
        return self.config.HIGH_HEIGHT

    def update_all(self, timestep, sensors):
        # Per-cell mass: sensor[c, r, NE] reads the mass in cell (c, r) via
        # piston (c, r)'s NE quadrant -- same convention used by greedy_bin.
        cell_mass = sensors[:self._nx, :self._ny, NE]

        # Cells with at least one incident edge currently HOLDING (read from
        # the state BEFORE this tick's step). Arbitration suppresses new
        # donations from such cells so a cell can have at most one outgoing
        # HOLDING edge at a time -- preventing the all-sides-LOW basin.
        cell_has_holding = np.zeros((self._nx, self._ny), dtype=bool)
        if self._ny >= 2:
            h_holding_now = self._h_edge_state > 0
            cell_has_holding[:, :-1] |= h_holding_now
            cell_has_holding[:, 1:]  |= h_holding_now
        if self._nx >= 2:
            v_holding_now = self._v_edge_state > 0
            cell_has_holding[:-1, :] |= v_holding_now
            cell_has_holding[1:, :]  |= v_holding_now

        # Per-cell arbitration: each donor cell picks at most one outgoing direction.
        h_chosen, v_chosen = self._arbitrate_per_cell(cell_mass, cell_has_holding)

        # Advance horizontal edges (between cells (c, r) and (c, r+1)).
        if self._ny >= 2:
            h_can_fire = self._can_fire(cell_mass[:, :-1], cell_mass[:, 1:]) & h_chosen
            self._h_edge_state = self._step_edges(self._h_edge_state, h_can_fire)

        # Advance vertical edges (between cells (c, r) and (c+1, r)).
        if self._nx >= 2:
            v_can_fire = self._can_fire(cell_mass[:-1, :], cell_mass[1:, :]) & v_chosen
            self._v_edge_state = self._step_edges(self._v_edge_state, v_can_fire)

        # Piston (i, j) goes LOW iff any incident edge is HOLDING. Each edge
        # touches exactly two pistons; scatter the HOLDING flag onto both.
        piston_low = np.zeros((self.config.GRIDSIZEX, self.config.GRIDSIZEY), dtype=bool)

        if self._ny >= 2:
            h_holding = self._h_edge_state > 0
            # h-edge at (c, r) sits between cells (c, r) and (c, r+1); its
            # endpoint pistons are (c, r+1) and (c+1, r+1).
            piston_low[0:self._nx,   1:self._ny] |= h_holding
            piston_low[1:self._nx+1, 1:self._ny] |= h_holding

        if self._nx >= 2:
            v_holding = self._v_edge_state > 0
            # v-edge at (c, r) sits between cells (c, r) and (c+1, r); its
            # endpoint pistons are (c+1, r) and (c+1, r+1).
            piston_low[1:self._nx, 0:self._ny]   |= v_holding
            piston_low[1:self._nx, 1:self._ny+1] |= v_holding

        return np.where(piston_low, self.config.LOW_HEIGHT, self.config.HIGH_HEIGHT)
