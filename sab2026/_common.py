import numpy as np


def make_clockwise_direction_map(n_rows, n_cols):
    dmap = np.empty((n_rows, n_cols), dtype='<U1')
    cy = (n_rows - 1) / 2.0
    cx = (n_cols - 1) / 2.0
    for j in range(n_rows):
        for i in range(n_cols):
            dy = j - cy
            dx = i - cx
            if abs(dy) > abs(dx):
                dmap[j, i] = 'E' if dy > 0 else 'W'
            elif abs(dx) > abs(dy):
                dmap[j, i] = 'N' if dx < 0 else 'S'
            elif dy > 0:   # tie, upper half: NW->E, NE->S
                dmap[j, i] = 'E' if dx < 0 else 'S'
            else:          # tie, lower half: SW->N, SE->W
                dmap[j, i] = 'N' if dx < 0 else 'W'
    return dmap


def compute_distance_metrics(ballsstates, config):
    """Per-timestep average pairwise distance and average distance to surface center.

    Uses 2D (x, y) positions. Returns two arrays of shape (T,).
    """
    cx = config.D_RODS * (config.GRIDSIZEX - 1) / 2.0
    cy = config.D_RODS * (config.GRIDSIZEY - 1) / 2.0
    center = np.array([cx, cy])

    T = len(ballsstates)
    pair_dist = np.zeros(T)
    center_dist = np.zeros(T)

    for t, balls in enumerate(ballsstates):
        xy = balls[:, :2]
        n = xy.shape[0]
        if n > 1:
            diffs = xy[:, None, :] - xy[None, :, :]
            d = np.linalg.norm(diffs, axis=-1)
            iu = np.triu_indices(n, k=1)
            pair_dist[t] = d[iu].mean()
        center_dist[t] = np.linalg.norm(xy - center, axis=1).mean()

    return pair_dist, center_dist
