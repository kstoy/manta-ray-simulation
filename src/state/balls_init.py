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
    R = rng.uniform(0.025, 0.075, size=N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    r = np.empty((N, 3), float)
    r[:, 0] = (np.tile(np.arange(config.GRIDSIZEX - 1), config.GRIDSIZEY - 1) + 0.5) * config.D_RODS
    r[:, 1] = (np.repeat(np.arange(config.GRIDSIZEY - 1), config.GRIDSIZEX - 1) + 0.5) * config.D_RODS

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


def random_positions(config, rodstate):
    """Balls at random x,y positions within the grid, fixed radii."""
    rng = np.random.default_rng()
    N = config.NBALL

    v = np.zeros((N, 3), float)
    w = np.zeros((N, 3), float)
    R = np.repeat([0.05], N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    r = np.empty((N, 3), float)
    margin = 0.1 * config.D_RODS
    r[:, 0] = rng.uniform(margin, config.D_RODS * (config.GRIDSIZEX - 1) - margin, size=N)
    r[:, 1] = rng.uniform(margin, config.D_RODS * (config.GRIDSIZEY - 1) - margin, size=N)

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


def center_cluster(config, rodstate):
    """All balls clustered near the grid center, random radii."""
    rng = np.random.default_rng()
    N = config.NBALL

    v = np.zeros((N, 3), float)
    w = np.zeros((N, 3), float)
    R = rng.uniform(0.025, 0.075, size=N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    cx = config.D_RODS * (config.GRIDSIZEX - 1) / 2.0
    cy = config.D_RODS * (config.GRIDSIZEY - 1) / 2.0
    spread = min(cx, cy) * 0.3

    r = np.empty((N, 3), float)
    margin = 0.1 * config.D_RODS
    r[:, 0] = rng.normal(cx, spread, size=N).clip(margin, config.D_RODS * (config.GRIDSIZEX - 1) - margin)
    r[:, 1] = rng.normal(cy, spread, size=N).clip(margin, config.D_RODS * (config.GRIDSIZEY - 1) - margin)

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


def outside_rectangle(config, rodstate):
    """All balls in a compact rectangular formation outside the grid."""
    N = config.NBALL

    v = np.zeros((N, 3), float)
    w = np.zeros((N, 3), float)
    R = np.repeat([0.1], N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    # Create compact rectangular formation
    spacing = 0.5  # Tight spacing between balls
    cols = int(np.ceil(np.sqrt(N)))

    # Position outside grid (below y=0, centered in x)
    grid_center_x = config.D_RODS * (config.GRIDSIZEX - 1) / 2.0
    rect_width = (cols - 1) * spacing
    start_x = grid_center_x - rect_width / 2.0
    start_y = -100.5  # Outside grid below y=0

    r = np.empty((N, 3), float)
    for i in range(N):
        col = i % cols
        row = i // cols
        r[i, 0] = start_x + col * spacing
        r[i, 1] = start_y - row * spacing

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


def perimeter(config, rodstate):
    """Balls distributed around the grid perimeter with approximately equal spacing."""
    N = config.NBALL

    v = np.zeros((N, 3), float)
    w = np.zeros((N, 3), float)
    R = np.repeat([config.BALL_RADIUS], N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    margin = 0.1 * config.D_RODS
    W = config.D_RODS * (config.GRIDSIZEX - 1) - 2 * margin
    H = config.D_RODS * (config.GRIDSIZEY - 1) - 2 * margin
    perimeter_length = 2 * (W + H)
    spacing = perimeter_length / N

    r = np.empty((N, 3), float)
    for i in range(N):
        s = i * spacing
        if s < W:
            r[i, 0] = margin + s
            r[i, 1] = margin
        elif s < W + H:
            r[i, 0] = margin + W
            r[i, 1] = margin + (s - W)
        elif s < 2 * W + H:
            r[i, 0] = margin + W - (s - W - H)
            r[i, 1] = margin + H
        else:
            r[i, 0] = margin
            r[i, 1] = margin + H - (s - 2 * W - H)

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


BALL_INIT_REGISTRY = {
    "grid_uniform": grid_uniform,
    "random": random_positions,
    "center_cluster": center_cluster,
    "outside_rectangle": outside_rectangle,
    "perimeter": perimeter,
}


def get_ball_init(name):
    """Get a ball initialization function by name."""
    if name not in BALL_INIT_REGISTRY:
        available = ", ".join(BALL_INIT_REGISTRY.keys())
        raise ValueError(f"Unknown ball init '{name}'. Available: {available}")
    return BALL_INIT_REGISTRY[name]


# --- Respawn position functions ---
# Each takes (config, rng) and returns (x, y) for a single ball respawn.

def _respawn_random(config, rng):
    margin = 0.1 * config.D_RODS
    x = rng.uniform(margin, config.D_RODS * (config.GRIDSIZEX - 1) - margin)
    y = rng.uniform(margin, config.D_RODS * (config.GRIDSIZEY - 1) - margin)
    return x, y


def _respawn_grid_uniform(config, rng):
    i = rng.integers(0, config.GRIDSIZEX - 1)
    j = rng.integers(0, config.GRIDSIZEY - 1)
    return (i + 0.5) * config.D_RODS, (j + 0.5) * config.D_RODS


def _respawn_center_cluster(config, rng):
    cx = config.D_RODS * (config.GRIDSIZEX - 1) / 2.0
    cy = config.D_RODS * (config.GRIDSIZEY - 1) / 2.0
    spread = min(cx, cy) * 0.3
    margin = 0.1 * config.D_RODS
    x = float(np.clip(rng.normal(cx, spread), margin, config.D_RODS * (config.GRIDSIZEX - 1) - margin))
    y = float(np.clip(rng.normal(cy, spread), margin, config.D_RODS * (config.GRIDSIZEY - 1) - margin))
    return x, y


def _respawn_southwest(config, rng):
    return config.D_RODS * 0.5, config.D_RODS * 0.5


RESPAWN_REGISTRY = {
    "grid_uniform": _respawn_grid_uniform,
    "random": _respawn_random,
    "center_cluster": _respawn_center_cluster,
    "outside_rectangle": _respawn_random,  # fallback: random on-surface position
    "perimeter": _respawn_random,  # fallback: random on-surface position
    "southwest": _respawn_southwest,
}


def get_respawn_position(name):
    """Get a respawn position function by BALL_INIT name."""
    if name not in RESPAWN_REGISTRY:
        available = ", ".join(RESPAWN_REGISTRY.keys())
        raise ValueError(f"Unknown respawn strategy '{name}'. Available: {available}")
    return RESPAWN_REGISTRY[name]
