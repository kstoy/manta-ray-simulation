import numpy as np


class BallsState:
    """
    r: (N,3) positions
    v: (N,3) linear velocities
    w: (N,3) angular velocities
    m: (N,)  masses
    R: (N,)  radii
    Inertia (solid sphere): I = 2/5 m R^2  (scalar per ball)
    """
    def __init__(self, rodstate, config):
        rng = np.random.default_rng()

        N = config.NBALL

        v = np.zeros((N, 3), float)
        w = np.zeros((N, 3), float)
        R = rng.uniform(0.05, 0.15, size=N)
        m = 2 * 4 / 3 * np.pi * np.power(R, 3)

        r = np.empty((N, 3), float)

        r[:, 0] = np.tile( np.arange(config.GRIDSIZEX - 1), config.GRIDSIZEY - 1) + 0.5
        r[:, 1] = np.repeat( np.arange( config.GRIDSIZEX - 1), config.GRIDSIZEY - 1) + 0.5

        for i in range(N):
            z, _, _ = rodstate.surfacejet(r[i, 0], r[i, 1])
            r[i, 2] = R[i] + z + 0.01

        for i in range(N):
            for j in range(i):
                dist = np.linalg.norm(r[i, :] - r[j, :])
                if dist < R[i] + R[j]:
                    # overlap, push up
                    r[i, 2] += (R[i] + R[j] - dist) + 0.05

        self.r = np.asarray(r, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.m = np.asarray(m, dtype=float)
        self.R = np.asarray(R, dtype=float)
        assert self.r.shape == self.v.shape == self.w.shape
        assert self.r.ndim == 2 and self.r.shape[1] == 3
        assert self.m.shape[0] == self.r.shape[0] and self.R.shape[0] == self.r.shape[0]
        self.N = self.r.shape[0]
        self.I = (2.0 / 5.0) * self.m * self.R * self.R      # solid sphere inertia
        self.inv_m = np.where(self.m > 0, 1.0 / self.m, 0.0)
        self.inv_I = np.where(self.I > 0, 1.0 / self.I, 0.0)
