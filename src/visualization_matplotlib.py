"""Matplotlib-based 3D visualization of the surface simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from src.physics import catenary as cat


def _compute_surface_grid(rods, config, resolution=20):
    """Compute a regular z-height grid over the full rod surface.

    Args:
        rods: Rod positions array of shape (GRIDSIZEX, GRIDSIZEY, 3).
        config: SimConfig instance.
        resolution: Number of sample points per module edge.

    Returns:
        X, Y, Z numpy arrays suitable for plot_surface.
    """
    nx = (config.GRIDSIZEX - 1) * resolution + 1
    ny = (config.GRIDSIZEY - 1) * resolution + 1

    X = np.linspace(0, (config.GRIDSIZEX - 1) * config.D, nx)
    Y = np.linspace(0, (config.GRIDSIZEY - 1) * config.D, ny)
    X2d, Y2d = np.meshgrid(X, Y, indexing='ij')
    Z = np.zeros_like(X2d)

    for i in range(config.GRIDSIZEX - 1):
        for j in range(config.GRIDSIZEY - 1):
            x0 = i * config.D
            y0 = j * config.D

            rod_sw = rods[i, j, 2]
            rod_se = rods[i + 1, j, 2]
            rod_nw = rods[i, j + 1, 2]
            rod_ne = rods[i + 1, j + 1, 2]

            cat_w = cat.findcatenaryparameters(config.LF, config.D, rod_sw, rod_nw)
            cat_e = cat.findcatenaryparameters(config.LF, config.D, rod_se, rod_ne)

            ix_start = i * resolution
            ix_end = (i + 1) * resolution + 1
            iy_start = j * resolution
            iy_end = (j + 1) * resolution + 1

            for ii in range(ix_start, ix_end):
                local_x = X[ii] - x0
                for jj in range(iy_start, iy_end):
                    local_y = Y[jj] - y0
                    h_w = cat.catenary(local_y, cat_w)
                    h_e = cat.catenary(local_y, cat_e)
                    cat_we = cat.findcatenaryparameters(config.LF, config.D, h_w, h_e)
                    Z[ii, jj] = cat.catenary(local_x, cat_we)

    return X2d, Y2d, Z


def animate_simulation(rodsstates, ballsstates, ballradiuses, config,
                       resolution=20, interval=50, save_path=None):
    """Show an animated matplotlib 3D visualization of the simulation.

    Args:
        rodsstates: List of rod arrays, one per timestep.
        ballsstates: List of ball position arrays (N, 3), one per timestep.
        ballradiuses: Array of ball radii.
        config: SimConfig instance.
        resolution: Surface grid points per module edge.
        interval: Milliseconds between animation frames.
        save_path: If provided, save the animation to this file (e.g. .mp4 or .gif).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute z-range across all frames for consistent axis limits
    z_min, z_max = np.inf, -np.inf
    for rods in rodsstates:
        z_min = min(z_min, rods[:, :, 2].min())
        z_max = max(z_max, rods[:, :, 2].max())
    for balls in ballsstates:
        z_min = min(z_min, balls[:, 2].min())
        z_max = max(z_max, balls[:, 2].max())
    z_pad = (z_max - z_min) * 0.2
    z_min -= z_pad
    z_max += z_pad

    x_max = (config.GRIDSIZEX - 1) * config.D
    y_max = (config.GRIDSIZEY - 1) * config.D

    # Initial frame
    X, Y, Z = _compute_surface_grid(rodsstates[0], config, resolution)
    surf = ax.plot_surface(X, Y, Z, color='green', alpha=0.6, edgecolor='none')

    balls_pos = ballsstates[0]
    sizes = (ballradiuses * 80) ** 2  # scale radii to marker area
    scatter = ax.scatter(balls_pos[:, 0], balls_pos[:, 1], balls_pos[:, 2],
                         c='red', s=sizes, depthshade=True)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = ax.set_title('Step 0')

    def update(frame):
        nonlocal surf, scatter
        # Remove old surface and scatter
        surf.remove()
        scatter.remove()

        # Recompute surface for this frame
        X, Y, Z = _compute_surface_grid(rodsstates[frame], config, resolution)
        surf = ax.plot_surface(X, Y, Z, color='green', alpha=0.6, edgecolor='none')

        # Recreate scatter with new ball positions
        balls_pos = ballsstates[frame]
        scatter = ax.scatter(balls_pos[:, 0], balls_pos[:, 1], balls_pos[:, 2],
                             c='red', s=sizes, depthshade=True)

        title.set_text(f'Step {frame}')
        return surf, scatter

    anim = animation.FuncAnimation(fig, update, frames=len(rodsstates),
                                   interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg')
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim
