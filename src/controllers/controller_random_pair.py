"""
Random Pair Controller

Each timestep, gather all edges currently eligible to fire — those past their
own REFRACTORY period and whose reserved rods don't intersect any other active
edge's reservation.  If at least one eligible edge exists, pick one uniformly
at random and fire it.  A fired edge holds both endpoint rods at LOW_HEIGHT
for RANDOM_PAIR_HOLD_STEPS, then enters a COOLDOWN of RANDOM_PAIR_COOLDOWN_STEPS
during which rods rise back to HIGH but the reservation is still held.  After
the reservation is released, the edge stays in a REFRACTORY phase for
RANDOM_PAIR_REFRACTORY_STEPS more timesteps before it can fire again — this
forces the system to keep rotating through different edges instead of locking
into a single absorbing matching.

Reservation set: a fired edge locks ALL six rods that are corners of either of
the two cells it borders, not just its two endpoint rods.  This blocks every
other edge of either bordering cell — including the opposite edge of the same
cell, which doesn't share a rod with the fired edge but would still produce
two simultaneous troughs on the same cell if allowed.

Sensors are unused; the controller is purely state-driven.
"""

import numpy as np
from src.controllers.controller_base import Controller


class ControllerRandomPair(Controller):
    def __init__(self, config):
        super().__init__(config)
        self.rng = np.random.default_rng()
        self.edges = self._enumerate_edges()
        self.edge_locks = [self._compute_lock(idx) for idx in range(len(self.edges))]
        self.hold_steps = config.RANDOM_PAIR_HOLD_STEPS
        self.cooldown_steps = config.RANDOM_PAIR_COOLDOWN_STEPS
        self.refractory_steps = config.RANDOM_PAIR_REFRACTORY_STEPS
        # Per-edge timestamps.  low_until: when the LOW phase ends (rods
        # commanded HIGH from then on).  busy_until: when the reservation
        # is released (other edges no longer blocked).  refractory_until:
        # when THIS edge becomes eligible to fire again.
        N = len(self.edges)
        self.low_until = np.full(N, -1, dtype=int)
        self.busy_until = np.full(N, 0, dtype=int)
        self.refractory_until = np.full(N, 0, dtype=int)

    def _enumerate_edges(self):
        """Adjacent-rod pairs whose shared edge separates two real cells."""
        gx, gy = self.config.GRIDSIZEX, self.config.GRIDSIZEY
        edges = []
        # Horizontal rod pairs: (i, j)-(i+1, j) separates cells (i, j-1) and (i, j)
        for i in range(gx - 1):
            for j in range(1, gy - 1):
                edges.append(((i, j), (i + 1, j)))
        # Vertical rod pairs: (i, j)-(i, j+1) separates cells (i-1, j) and (i, j)
        for i in range(1, gx - 1):
            for j in range(gy - 1):
                edges.append(((i, j), (i, j + 1)))
        return edges

    def _cells_for_edge(self, idx):
        """Indices of the two cells the edge with index `idx` separates."""
        (a, b) = self.edges[idx]
        if a[0] == b[0]:                          # vertical: same x
            i, j = a[0], min(a[1], b[1])
            return [(i - 1, j), (i, j)]
        else:                                     # horizontal: same y
            i, j = min(a[0], b[0]), a[1]
            return [(i, j - 1), (i, j)]

    def _compute_lock(self, idx):
        """All rod positions that this edge reserves while active (the four
        corners of each of its two bordering cells; six unique rods)."""
        rods = set()
        for (cx, cy) in self._cells_for_edge(idx):
            for dx in (0, 1):
                for dy in (0, 1):
                    rods.add((cx + dx, cy + dy))
        return frozenset(rods)

    def update(self, i, j, timestep, sensors):
        # Per-rod path is unused; update_all() handles all rods together.
        return self.config.HIGH_HEIGHT

    def update_all(self, timestep, sensors):
        desired = np.full(
            (self.config.GRIDSIZEX, self.config.GRIDSIZEY),
            self.config.HIGH_HEIGHT,
        )

        # Rods reserved by edges currently in LOW or COOLDOWN phase.
        busy_rods = set()
        for idx in np.nonzero(timestep < self.busy_until)[0]:
            busy_rods |= self.edge_locks[idx]

        # All currently eligible edges: past refractory AND not blocked by an
        # active edge's reservation.
        ready = self.refractory_until <= timestep
        eligible = [
            idx for idx in np.nonzero(ready)[0]
            if not (self.edge_locks[idx] & busy_rods)
        ]

        # Fire one randomly-chosen eligible edge.
        if eligible:
            chosen = int(self.rng.choice(eligible))
            self.low_until[chosen] = timestep + self.hold_steps
            self.busy_until[chosen] = timestep + self.hold_steps + self.cooldown_steps
            self.refractory_until[chosen] = self.busy_until[chosen] + self.refractory_steps

        # Drive rods of currently-low edges to LOW_HEIGHT (cooldown-phase
        # edges remain at the default HIGH_HEIGHT so they rise back).
        for idx in np.nonzero(timestep < self.low_until)[0]:
            rod_a, rod_b = self.edges[idx]
            desired[rod_a] = self.config.LOW_HEIGHT
            desired[rod_b] = self.config.LOW_HEIGHT

        return desired
