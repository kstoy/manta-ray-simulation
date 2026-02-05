"""Ball initialization functions.

Each function takes (config, rodstate) and returns (r, v, w, m, R) arrays.
"""

import numpy as np


def _resolve_overlaps(r, R):
    """Push overlapping balls upward to resolve collisions."""
    N = len(R)
    for i in range(N):
        for j in range(i):
            dist = np.linalg.norm(r[i, :] - r[j, :])
            if dist < R[i] + R[j]:
                r[i, 2] += (R[i] + R[j] - dist) + 0.05


def _set_z_from_surface(r, R, rodstate):
    """Set z positions so balls rest on the surface."""
    for i in range(len(R)):
        z, _, _ = rodstate.surfacejet(r[i, 0], r[i, 1])
        r[i, 2] = R[i] + z + 0.01


def grid_uniform(config, rodstate):
    """One ball per grid cell at cell centers, random radii."""
    rng = np.random.default_rng()
    N = config.NBALL

    v = np.zeros((N, 3), float)
    w = np.zeros((N, 3), float)
    R = rng.uniform(0.05, 0.15, size=N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    r = np.empty((N, 3), float)
    r[:, 0] = np.tile(np.arange(config.GRIDSIZEX - 1), config.GRIDSIZEY - 1) + 0.5
    r[:, 1] = np.repeat( np.arange( config.GRIDSIZEY - 1 ), config.GRIDSIZEX - 1 )

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


def random_positions(config, rodstate):
    """Balls at random x,y positions within the grid, random radii."""
    rng = np.random.default_rng()
    N = config.NBALL

    v = np.zeros((N, 3), float)
    w = np.zeros((N, 3), float)
    R = rng.uniform(0.05, 0.15, size=N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    r = np.empty((N, 3), float)
    r[:, 0] = rng.uniform(0.5, config.D * (config.GRIDSIZEX - 1) - 0.5, size=N)
    r[:, 1] = rng.uniform(0.5, config.D * (config.GRIDSIZEY - 1) - 0.5, size=N)

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


def center_cluster(config, rodstate):
    """All balls clustered near the grid center, random radii."""
    rng = np.random.default_rng()
    N = config.NBALL

    v = np.zeros((N, 3), float)
    w = np.zeros((N, 3), float)
    R = rng.uniform(0.05, 0.15, size=N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    cx = config.D * (config.GRIDSIZEX - 1) / 2.0
    cy = config.D * (config.GRIDSIZEY - 1) / 2.0
    spread = min(cx, cy) * 0.3

    r = np.empty((N, 3), float)
    r[:, 0] = rng.normal(cx, spread, size=N).clip(0.5, config.D * (config.GRIDSIZEX - 1) - 0.5)
    r[:, 1] = rng.normal(cy, spread, size=N).clip(0.5, config.D * (config.GRIDSIZEY - 1) - 0.5)

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


BALL_INIT_REGISTRY = {
    "grid_uniform": grid_uniform,
    "random": random_positions,
    "center_cluster": center_cluster,
}


def get_ball_init(name):
    """Get a ball initialization function by name."""
    if name not in BALL_INIT_REGISTRY:
        available = ", ".join(BALL_INIT_REGISTRY.keys())
        raise ValueError(f"Unknown ball init '{name}'. Available: {available}")
    return BALL_INIT_REGISTRY[name]
