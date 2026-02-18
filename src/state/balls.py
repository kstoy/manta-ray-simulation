import numpy as np

from src.state.balls_init import get_ball_init


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
        init_fn = get_ball_init(config.BALL_INIT)
        r, v, w, m, R = init_fn(config, rodstate)

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
