"""
Stochastic Assembly Controller (drift + diffusion self-assembly).

Forms a target pattern (the BIT_MAP) by stochastic self-assembly rather than
directed routing.  Placement is decided per cell by a "stay" probability; an
ejected ball moves in a biased-random direction; and every cell runs an explicit
catch -> rest -> decide -> eject state machine so the rods always fully actuate
(no chatter / shaking on the spot).

Placement (where balls want to rest), Ising/Metropolis style:

    s(c) =  h * [c is a pattern cell]                # pinning field: hold the shape
          - q * (# adjacent MISPLACED balls)          # penalty: go porous near wrong balls
          - p * [c is an off-pattern cell]            # off-pattern is unstable
    P_stay = sigmoid(beta_temp * s)

  A "misplaced" ball sits on a 0-cell.  Pinning (h) gives every pattern cell
  baseline stability independent of neighbour count, so thin 1-wide strokes hold.
  The penalty (q) destabilises pattern cells next to misplaced balls, making walls
  porous so balls trapped inside closed loops (e.g. the "O") can leak out.

Movement (how an ejected ball moves) -- softmax over the 4 neighbours, biased by
a precomputed field built from the known map:

    bias = gamma * CCW_circulation  +  alpha * attract_gradient
    w(dir) ~ exp(beta_drift * (bias . dir))

Enclosed holes (0-cells not reachable from the border) are forbidden: P_stay = 0
with a strong outward escape bias.

Cell state machine (fixes the rod chatter) -- a ball in a cell is always in one
of CATCH / REST / EJECT, each held for a fixed number of steps so the rods reach
their commanded limit before anything is re-decided:

  EMPTY  : no ball; rods stay raised.
  CATCH  : a ball just arrived; briefly lower the FAR edge (its incoming
           direction) to pull it past the cell boundary so it can't rattle back
           (the receiver-side hand-off).  -> REST.
  REST   : raise all four edges and hold REST_HOLD steps -- long enough for the
           rods to reach maximum height and the ball to settle in the cell.  When
           the hold expires, run the stochastic stay/move decision.
  EJECT  : the cell chose to move the ball; lower the whole edge (both rods) in
           the chosen direction and hold EJECT steps so the ball fully crosses.
           The receiving cell then catches it.

Annealing over the run (t = timestep / MAXSIMULATIONSTEPS): beta_temp cold late
(lock in), beta_drift directed->diffusive (fill then heal), q porous->rigid.

The pattern is provided as the BIT_MAP (in the DIRECTION_MAP slot of an
experiment): a 2D array of '0'/'1' strings shaped (GRIDSIZEY-1, GRIDSIZEX-1).
"""

import numpy as np
from collections import deque

from src.controllers.controller_base import Controller
from src.config import NE, NW, SW, SE


class ControllerStochasticAssembly(Controller):
    # Cell states.
    EMPTY, CATCH, REST, EJECT = 0, 1, 2, 3

    # Cell-space step (d_cx, d_cy) per direction.  N = +cy, E = +cx.
    DIR_STEP = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
    DIRS = ('N', 'S', 'E', 'W')
    OPP = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, config, bit_map):
        super().__init__(config)

        self.pattern = (np.asarray(bit_map).astype(str) == '1')
        self.ny, self.nx = self.pattern.shape

        self.rng = np.random.default_rng(int(getattr(config, 'SA_SEED', 0)))

        # Single-ball mass m = (8*pi/3) R^3 (see balls_init), used to count balls
        # per cell.  A cell counts as occupied above a fraction of one ball, and
        # an eject ends once one ball's worth has entered the destination.
        R = float(getattr(config, 'BALL_RADIUS', 0.05))
        self.m_ball = 2.0 * 4.0 / 3.0 * np.pi * R ** 3
        self.present_mass = self.m_ball * float(getattr(config, 'SA_PRESENT_FRACTION', 0.5))
        self.transfer_mass = self.m_ball * float(getattr(config, 'SA_TRANSFER_FRACTION', 0.5))

        # Placement knobs (q is annealed at runtime).
        self.h = float(getattr(config, 'SA_H', 3.0))
        self.p_off = float(getattr(config, 'SA_P_OFF', 2.0))

        # Annealing endpoints.
        self.beta_start = float(getattr(config, 'SA_BETA_START', 0.5))
        self.beta_end = float(getattr(config, 'SA_BETA_END', 4.0))
        self.bdrift_start = float(getattr(config, 'SA_BETADRIFT_START', 3.0))
        self.bdrift_end = float(getattr(config, 'SA_BETADRIFT_END', 0.5))
        self.q_start = float(getattr(config, 'SA_Q_START', 2.0))
        self.q_end = float(getattr(config, 'SA_Q_END', 0.5))

        # Drift composition.
        self.alpha = float(getattr(config, 'SA_ALPHA', 0.5))   # attract weight (small)
        self.gamma = float(getattr(config, 'SA_GAMMA', 1.0))   # circulation weight
        self.escape_strength = float(getattr(config, 'SA_ESCAPE_STRENGTH', 4.0))
        self.forbid_holes = bool(getattr(config, 'SA_FORBID_HOLES', True))

        # State-machine timing (in simulation steps).  REST_HOLD must be long
        # enough for a rod to travel LOW<->HIGH under the P-control gain K:
        # n ~ log(eps)/log(1-K) (e.g. K=0.2 -> ~20 steps to settle).
        self.rest_hold = int(getattr(config, 'SA_REST_HOLD_STEPS', 25))
        self.catch_steps = int(getattr(config, 'SA_CATCH_STEPS', 5))
        self.eject_steps = int(getattr(config, 'SA_EJECT_STEPS', 12))

        self._build_static_fields()

        # Per-cell state machine, indexed [cy, cx].
        self.state = np.full((self.ny, self.nx), self.EMPTY, dtype=int)
        self.timer = np.zeros((self.ny, self.nx), dtype=int)        # step the phase ends
        self.dir = np.full((self.ny, self.nx), 'I', dtype='U1')     # CATCH: incoming; EJECT: outgoing
        self.incoming_dir = np.full((self.ny, self.nx), 'I', dtype='U1')  # announced by an ejecting neighbour
        self.occ_prev = np.zeros((self.ny, self.nx), dtype=bool)

    # ------------------------------------------------------------------ setup

    def _bfs(self, seeds, passable=None):
        """Unweighted distance over the cell grid from seed cells (-1 = unreached)."""
        dist = np.full((self.ny, self.nx), -1, dtype=int)
        dq = deque()
        for (cy, cx) in seeds:
            dist[cy, cx] = 0
            dq.append((cy, cx))
        while dq:
            cy, cx = dq.popleft()
            for dcy, dcx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ncy, ncx = cy + dcy, cx + dcx
                if 0 <= ncy < self.ny and 0 <= ncx < self.nx and dist[ncy, ncx] == -1:
                    if passable is not None and not passable[ncy, ncx]:
                        continue
                    dist[ncy, ncx] = dist[cy, cx] + 1
                    dq.append((ncy, ncx))
        return dist

    @staticmethod
    def _normalize(vx, vy):
        mag = np.hypot(vx, vy)
        safe = mag > 1e-9
        ux = np.where(safe, vx / np.where(safe, mag, 1.0), 0.0)
        uy = np.where(safe, vy / np.where(safe, mag, 1.0), 0.0)
        return ux, uy

    def _downhill_unit(self, dist):
        """Unit field pointing downhill on a distance field; (vcx, vcy), N=+cy, E=+cx."""
        d = dist.astype(float)
        d[dist < 0] = (dist.max() + 1) if dist.max() >= 0 else 0.0
        g_cy, g_cx = np.gradient(d)
        return self._normalize(-g_cx, -g_cy)

    def _build_static_fields(self):
        ny, nx = self.ny, self.nx
        pattern_cells = [(cy, cx) for cy in range(ny) for cx in range(nx)
                         if self.pattern[cy, cx]]

        dist_pat = self._bfs(pattern_cells) if pattern_cells else np.zeros((ny, nx), int)
        att_cx, att_cy = self._downhill_unit(dist_pat)
        circ_cx, circ_cy = -att_cy, att_cx   # rotate attract +90 deg CCW

        non_pattern = ~self.pattern
        border_seeds = [(cy, cx) for cy in range(ny) for cx in range(nx)
                        if (cy in (0, ny - 1) or cx in (0, nx - 1)) and non_pattern[cy, cx]]
        reach = self._bfs(border_seeds, passable=non_pattern) >= 0
        self.forbidden = non_pattern & ~reach if self.forbid_holes else np.zeros((ny, nx), bool)

        free_seeds = [(cy, cx) for cy in range(ny) for cx in range(nx)
                      if reach[cy, cx] and non_pattern[cy, cx]]
        if free_seeds:
            esc_cx, esc_cy = self._downhill_unit(self._bfs(free_seeds))
        else:
            esc_cx, esc_cy = np.zeros((ny, nx)), np.zeros((ny, nx))

        bias_cx = self.gamma * circ_cx + self.alpha * att_cx
        bias_cy = self.gamma * circ_cy + self.alpha * att_cy
        if self.forbidden.any():
            bias_cx = np.where(self.forbidden, self.escape_strength * esc_cx, bias_cx)
            bias_cy = np.where(self.forbidden, self.escape_strength * esc_cy, bias_cy)
        self.bias_cx, self.bias_cy = bias_cx, bias_cy

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _lerp(t, a, b):
        return a + (b - a) * t

    def _edge_rods(self, cx, cy, d):
        """The two rods (i=x, j=y) on the cell's edge in direction d."""
        if d == 'E':
            return [(cx + 1, cy), (cx + 1, cy + 1)]
        if d == 'W':
            return [(cx, cy), (cx, cy + 1)]
        if d == 'N':
            return [(cx, cy + 1), (cx + 1, cy + 1)]
        if d == 'S':
            return [(cx, cy), (cx + 1, cy)]
        return []

    def _sample_direction(self, cy, cx, beta_drift):
        """Softmax over in-bounds neighbours, biased by the static field."""
        bx, by = self.bias_cx[cy, cx], self.bias_cy[cy, cx]
        dirs, aligns = [], []
        for d in self.DIRS:
            dcx, dcy = self.DIR_STEP[d]
            ncy, ncx = cy + dcy, cx + dcx
            if 0 <= ncy < self.ny and 0 <= ncx < self.nx:
                dirs.append(d)
                aligns.append(bx * dcx + by * dcy)
        if not dirs:
            return 'I'
        a = np.asarray(aligns)
        w = np.exp(beta_drift * (a - a.max()))
        w /= w.sum()
        return dirs[int(self.rng.choice(len(dirs), p=w))]

    def _stay_probability(self, cy, cx, occupied, q, beta_temp):
        if self.forbidden[cy, cx]:
            return 0.0
        is_pat = self.pattern[cy, cx]
        misplaced = 0
        for d in self.DIRS:
            dcx, dcy = self.DIR_STEP[d]
            ncy, ncx = cy + dcy, cx + dcx
            if 0 <= ncy < self.ny and 0 <= ncx < self.nx:
                if occupied[ncy, ncx] and not self.pattern[ncy, ncx]:
                    misplaced += 1
        s = self.h * is_pat - q * misplaced - self.p_off * (not is_pat)
        return 1.0 / (1.0 + np.exp(-beta_temp * s))

    # ------------------------------------------------------------- per-step

    def update(self, i, j, timestep, sensors):
        # update_all() drives the controller; this satisfies the abstract API.
        return self.config.HIGH_HEIGHT

    def update_all(self, timestep, sensors):
        cfg = self.config
        gx, gy = cfg.GRIDSIZEX, cfg.GRIDSIZEY

        t = min(1.0, timestep / max(1, cfg.MAXSIMULATIONSTEPS - 1))
        beta_temp = self._lerp(t, self.beta_start, self.beta_end)
        beta_drift = self._lerp(t, self.bdrift_start, self.bdrift_end)
        q = self._lerp(t, self.q_start, self.q_end)

        # Exact per-cell ball mass = sum of the cell's four corner-quadrant
        # sensor contributions (bilinear weights partition each ball's mass), so
        # two balls in one cell read ~2*m_ball regardless of their position.
        nx, ny = self.nx, self.ny
        cell_mass = (sensors[0:nx,     0:ny,     NE]
                     + sensors[1:nx + 1, 0:ny,     NW]
                     + sensors[0:nx,     1:ny + 1, SE]
                     + sensors[1:nx + 1, 1:ny + 1, SW]).T   # -> [cy, cx]
        occupied = cell_mass > self.present_mass

        for cy in range(self.ny):
            for cx in range(self.nx):
                if not occupied[cy, cx]:
                    self.state[cy, cx] = self.EMPTY
                    continue

                st = self.state[cy, cx]

                if st == self.EMPTY:
                    # Ball just arrived. If a neighbour announced its direction,
                    # pull it in with a short far-edge catch pulse; else rest.
                    d = self.incoming_dir[cy, cx]
                    if d in self.DIR_STEP and self.catch_steps > 0:
                        self.state[cy, cx] = self.CATCH
                        self.dir[cy, cx] = d
                        self.timer[cy, cx] = timestep + self.catch_steps
                    else:
                        self.state[cy, cx] = self.REST
                        self.timer[cy, cx] = timestep + self.rest_hold
                    self.incoming_dir[cy, cx] = 'I'

                elif st == self.CATCH:
                    if timestep >= self.timer[cy, cx]:
                        self.state[cy, cx] = self.REST
                        self.timer[cy, cx] = timestep + self.rest_hold

                elif st == self.REST:
                    if timestep >= self.timer[cy, cx]:
                        self._decide(cy, cx, occupied, q, beta_temp, beta_drift, timestep)

                elif st == self.EJECT:
                    # Raise the edge as soon as ONE ball's worth of mass has
                    # entered the destination, so exactly one ball transfers even
                    # when several share the source cell (a trailing ball can't
                    # follow through the closing edge).  The receiver's far-edge
                    # pull completes the crossing.  The timer is a fallback if the
                    # ball never crosses (blocked).
                    d = self.dir[cy, cx]
                    dcx, dcy = self.DIR_STEP.get(d, (0, 0))
                    tcy, tcx = cy + dcy, cx + dcx
                    crossed = (0 <= tcy < self.ny and 0 <= tcx < self.nx
                               and cell_mass[tcy, tcx] >= self.transfer_mass)
                    if crossed or timestep >= self.timer[cy, cx]:
                        self.state[cy, cx] = self.REST
                        self.timer[cy, cx] = timestep + self.rest_hold

        desired = self._rod_heights(gx, gy)
        self.occ_prev = occupied
        return desired

    def _decide(self, cy, cx, occupied, q, beta_temp, beta_drift, timestep):
        """REST hold expired: stay (re-rest) or eject in a biased direction."""
        if self.rng.random() < self._stay_probability(cy, cx, occupied, q, beta_temp):
            self.timer[cy, cx] = timestep + self.rest_hold
            return
        d = self._sample_direction(cy, cx, beta_drift)
        dcx, dcy = self.DIR_STEP.get(d, (0, 0))
        tcy, tcx = cy + dcy, cx + dcx
        free = d in self.DIR_STEP and 0 <= tcy < self.ny and 0 <= tcx < self.nx \
            and not occupied[tcy, tcx]
        if free:
            self.state[cy, cx] = self.EJECT
            self.dir[cy, cx] = d
            self.timer[cy, cx] = timestep + self.eject_steps
            self.incoming_dir[tcy, tcx] = d   # announce to the receiving cell
        else:
            self.timer[cy, cx] = timestep + self.rest_hold   # blocked -> keep resting

    def _rod_heights(self, gx, gy):
        """EJECT lowers the chosen edge; CATCH lowers the far (incoming) edge."""
        desired = np.full((gx, gy), self.config.HIGH_HEIGHT)
        for cy in range(self.ny):
            for cx in range(self.nx):
                st = self.state[cy, cx]
                if st == self.EJECT or st == self.CATCH:
                    for ri, rj in self._edge_rods(cx, cy, self.dir[cy, cx]):
                        desired[ri, rj] = self.config.LOW_HEIGHT
        return desired
