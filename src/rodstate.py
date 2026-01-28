import numpy as np

from src.controllers.squarecontroller_nonedeterministic_push import SquareControllerPush
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
        self.controller = SquareControllerPush(config)

    def settimestep(self, timestep):
        self.timestep = timestep

    def update(self):
        for i in range(self.config.GRIDSIZEX):
            for j in range(self.config.GRIDSIZEY):
                desired = self.controller.update(i, j, self.timestep, self.sensors[i][j])
                self.rods[i][j][2] += self.config.K * (desired - self.rods[i][j][2])

    def positiontoindex(self, x, y):
        return np.array([int(x / self.config.D), int(y / self.config.D)])

    def surfacejet(self, x, y):
        if (x < 0.0 or x > self.config.D * (self.config.GRIDSIZEX - 1)
                or y < 0.0 or y > self.config.D * (self.config.GRIDSIZEY - 1)):
            return np.array([-2.0, 0.0, 0.0])

        x_idx, y_idx = self.positiontoindex(x, y)
        x_local = x - float(x_idx * self.config.D)
        y_local = y - float(y_idx * self.config.D)

        rodheights = np.array([
            self.rods[x_idx][y_idx][2],
            self.rods[x_idx + 1][y_idx][2],
            self.rods[x_idx][y_idx + 1][2],
            self.rods[x_idx + 1][y_idx + 1][2]
        ])
        return catenarysurface.jet1(x_local, y_local, rodheights, self.config.D, self.config.LF)
