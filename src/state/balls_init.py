"""Ball initialization functions.

Each function takes (config, rodstate) and returns (r, v, w, m, R) arrays.
"""

import numpy as np


def seeded_rng(config, stream=0):
    """Return a NumPy Generator seeded from config.SEED.

    If config.SEED is None the generator is non-deterministic. Otherwise the
    result is deterministic and independent per `stream` index, so callers that
    need separate draws (e.g. initial placement vs. respawn) get uncorrelated
    streams from the same master seed.
    """
    seed = getattr(config, "SEED", None)
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def _perimeter_point(s, margin, W, H):
    """Map an arc-length position s in [0, 2*(W+H)) to an (x, y) on the grid
    perimeter inset by `margin`. Shared by `perimeter_random` and its respawn
    counterpart."""
    if s < W:
        return margin + s, margin
    elif s < W + H:
        return margin + W, margin + (s - W)
    elif s < 2 * W + H:
        return margin + W - (s - W - H), margin + H
    else:
        return margin, margin + H - (s - 2 * W - H)


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


def perimeter_random(config, rodstate):
    """Balls at random positions along the grid perimeter, fixed radii.

    Like `perimeter` but each ball's arc-length position is sampled uniformly at
    random (seeded via config.SEED) instead of evenly spaced, so repeated runs
    with different seeds produce distinct starting layouts.

    No two balls are placed in the same grid cell: each sampled position is
    rejected and resampled if its cell already holds a previously placed ball.
    The perimeter spans 2*(nx) + 2*(ny) - 4 border cells, comfortably more than
    NBALL for this experiment, so rejection sampling terminates quickly.
    """
    rng = seeded_rng(config, stream=0)
    N = config.NBALL

    v = np.zeros((N, 3), float)
    w = np.zeros((N, 3), float)
    R = np.repeat([config.BALL_RADIUS], N)
    m = 2 * 4 / 3 * np.pi * np.power(R, 3)

    margin = 0.1 * config.D_RODS
    W = config.D_RODS * (config.GRIDSIZEX - 1) - 2 * margin
    H = config.D_RODS * (config.GRIDSIZEY - 1) - 2 * margin
    perim = 2 * (W + H)
    inv = 1.0 / config.D_RODS
    max_attempts = 1000

    r = np.empty((N, 3), float)
    occupied = set()
    for i in range(N):
        x, y = _perimeter_point(rng.uniform(0.0, perim), margin, W, H)
        for _ in range(max_attempts):
            cell = (int(np.floor(x * inv)), int(np.floor(y * inv)))
            if cell not in occupied:
                occupied.add(cell)
                break
            x, y = _perimeter_point(rng.uniform(0.0, perim), margin, W, H)
        r[i, 0], r[i, 1] = x, y

    _set_z_from_surface(r, R, rodstate)
    _resolve_overlaps(r, R)

    return r, v, w, m, R


BALL_INIT_REGISTRY = {
    "grid_uniform": grid_uniform,
    "random": random_positions,
    "center_cluster": center_cluster,
    "outside_rectangle": outside_rectangle,
    "perimeter": perimeter,
    "perimeter_random": perimeter_random,
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


def _respawn_perimeter_random(config, rng):
    """Respawn at a uniformly random point along the grid perimeter."""
    margin = 0.1 * config.D_RODS
    W = config.D_RODS * (config.GRIDSIZEX - 1) - 2 * margin
    H = config.D_RODS * (config.GRIDSIZEY - 1) - 2 * margin
    return _perimeter_point(rng.uniform(0.0, 2 * (W + H)), margin, W, H)


RESPAWN_REGISTRY = {
    "grid_uniform": _respawn_grid_uniform,
    "random": _respawn_random,
    "center_cluster": _respawn_center_cluster,
    "outside_rectangle": _respawn_random,  # fallback: random on-surface position
    "perimeter": _respawn_random,  # fallback: random on-surface position
    "southwest": _respawn_southwest,
    "perimeter_random": _respawn_perimeter_random,
}


def get_respawn_position(name):
    """Get a respawn position function by BALL_INIT name."""
    if name not in RESPAWN_REGISTRY:
        available = ", ".join(RESPAWN_REGISTRY.keys())
        raise ValueError(f"Unknown respawn strategy '{name}'. Available: {available}")
    return RESPAWN_REGISTRY[name]
