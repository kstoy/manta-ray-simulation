import numpy as np

from src.controllers import get_controller
from src.physics import catenarysurface


class RodsState:
    def __init__(self, config):
        self.config = config
        self.rods = np.empty((config.GRIDSIZEX, config.GRIDSIZEY, 3), dtype=float)

        i_indices, j_indices = np.meshgrid(
            np.arange(config.GRIDSIZEX), np.arange(config.GRIDSIZEY), indexing='ij'
        )

        self.rods[:, :, 0] = i_indices * config.D
        self.rods[:, :, 1] = j_indices * config.D
        self.rods[:, :, 2] = 1.0

        self.sensors = np.full((config.GRIDSIZEX, config.GRIDSIZEY, 4), False, dtype=float)
        self.timestep = 0.0
        self.controller = get_controller(config.CONTROLLER, config)

    def settimestep(self, timestep):
        self.timestep = timestep

    def update(self):
        # Vectorized controller update for all rods at once
        desired = self.controller.update_all(self.timestep, self.sensors)

        # Vectorized P-control: height += K * (desired - current)
        self.rods[:, :, 2] += self.config.K * (desired - self.rods[:, :, 2])

    def positiontoindex(self, x, y):
        return np.array([int(x / self.config.D), int(y / self.config.D)])

    def surfacejet(self, x, y):
        if (x < 0.0 or x > self.config.D * (self.config.GRIDSIZEX - 1)
                or y < 0.0 or y > self.config.D * (self.config.GRIDSIZEY - 1)):
            return (-2.0, 0.0, 0.0)

        x_idx = int(x / self.config.D)
        y_idx = int(y / self.config.D)

        # Clamp indices to ensure +1 access is valid (need room for 4 corners)
        if x_idx > self.config.GRIDSIZEX - 2:
            x_idx = self.config.GRIDSIZEX - 2
        if y_idx > self.config.GRIDSIZEY - 2:
            y_idx = self.config.GRIDSIZEY - 2

        x_local = x - x_idx * self.config.D
        y_local = y - y_idx * self.config.D

        rodheights = (
            self.rods[x_idx, y_idx, 2],
            self.rods[x_idx + 1, y_idx, 2],
            self.rods[x_idx, y_idx + 1, 2],
            self.rods[x_idx + 1, y_idx + 1, 2],
        )
        return catenarysurface.jet1(x_local, y_local, rodheights, self.config.D, self.config.LF)

    def surfacejet_batch(self, xs, ys):
        """Vectorized surface height + gradients for N query points.

        Args:
            xs, ys: 1-D arrays of length N (world x, y coordinates)

        Returns:
            (z_s, dfx, dfy) — three arrays of length N.
            Out-of-bounds points get z_s = -2.0, dfx = dfy = 0.0.
        """
        D = self.config.D
        gx_max = D * (self.config.GRIDSIZEX - 1)
        gy_max = D * (self.config.GRIDSIZEY - 1)

        N = len(xs)
        z_s = np.full(N, -2.0)
        dfx = np.zeros(N)
        dfy = np.zeros(N)

        # In-bounds mask
        valid = (xs >= 0.0) & (xs <= gx_max) & (ys >= 0.0) & (ys <= gy_max)
        if not np.any(valid):
            return z_s, dfx, dfy

        xv = xs[valid]
        yv = ys[valid]

        xi = np.floor(xv / D).astype(int)
        yi = np.floor(yv / D).astype(int)
        np.clip(xi, 0, self.config.GRIDSIZEX - 2, out=xi)
        np.clip(yi, 0, self.config.GRIDSIZEY - 2, out=yi)

        x_local = xv - xi * D
        y_local = yv - yi * D

        rod_00 = self.rods[xi, yi, 2]
        rod_10 = self.rods[xi + 1, yi, 2]
        rod_01 = self.rods[xi, yi + 1, 2]
        rod_11 = self.rods[xi + 1, yi + 1, 2]

        f, dx, dy = catenarysurface.jet1_batch(
            x_local, y_local, rod_00, rod_10, rod_01, rod_11, D, self.config.LF
        )
        z_s[valid] = f
        dfx[valid] = dx
        dfy[valid] = dy
        return z_s, dfx, dfy
