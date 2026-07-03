"""Visualization modules for the surface simulation."""

import numpy as np


def compute_cell_vertex_colors(weight_frame, n_cells_x, n_cells_y,
                                nx, ny, resolution, target_weight):
    """Per-surface-vertex RGB color array shaped (nx*ny, 3) float32.

    weight_frame: (n_cells_x, n_cells_y) per-cell weight, or None for default white.
    Vertex flat-index matches the surface-mesh builders: i * ny + j.

    Colour ramp: white at 0, ramping through pink toward dark red at 1.5 × target.
    When weight_frame is None the result is all-white so existing demos render
    identically to before. Used by both the interactive viewer and the video
    exporter so colouring is consistent.
    """
    if weight_frame is None or target_weight <= 0:
        return np.ones((nx * ny, 3), dtype=np.float32)
    ii = np.minimum(np.arange(nx) // resolution, n_cells_x - 1)
    jj = np.minimum(np.arange(ny) // resolution, n_cells_y - 1)
    weights = weight_frame[ii[:, None], jj[None, :]]                   # (nx, ny)
    t = np.minimum(weights / target_weight, 1.5) / 1.5                  # 0..1
    colors = np.stack([1.0 - t * 0.3, 1.0 - t * 0.8, 1.0 - t * 0.8],
                      axis=-1).astype(np.float32)                       # (nx, ny, 3)
    return colors.reshape(nx * ny, 3)


def colors_from_radii(radii, cmap_name='coolwarm'):
    """Map a per-ball radius array to RGB colors keyed to deviation from the mean.

    A diverging colormap is centred on the mean radius: below-average balls
    shade cool/blue, above-average shade warm/red, near-mean balls are near-white.
    The colour saturation also encodes magnitude, so heavier balls show as deeper
    red and lighter balls as deeper blue.

    Returns an (N, 3) float32 array in [0, 1].
    """
    R = np.asarray(radii, dtype=float)
    if R.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mean = R.mean()
    dev = R - mean
    max_dev = float(np.max(np.abs(dev)))
    if max_dev > 0:
        # Map dev ∈ [-max_dev, +max_dev] to norm ∈ [0, 1], with 0.5 at the mean.
        norm = 0.5 + 0.5 * (dev / max_dev)
    else:
        norm = np.full_like(R, 0.5)
    try:
        import matplotlib as mpl
        # matplotlib.cm.get_cmap was removed in 3.9; prefer the modern API and
        # fall back to the old one for older matplotlib.
        try:
            cmap = mpl.colormaps[cmap_name]
        except AttributeError:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(cmap_name)
        rgba = cmap(norm)
        return rgba[:, :3].astype(np.float32)
    except ImportError:
        # Fallback: explicit blue→white→red diverging gradient.
        # norm < 0.5 → cool (blue), norm > 0.5 → warm (red), 0.5 → white.
        t = 2.0 * norm - 1.0           # ∈ [-1, 1]
        warm = np.clip(t, 0.0, 1.0)    # red strength
        cool = np.clip(-t, 0.0, 1.0)   # blue strength
        rgb = np.stack([1.0 - cool, 1.0 - cool - warm, 1.0 - warm], axis=1)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)
