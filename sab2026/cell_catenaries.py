"""Single-cell visualization showing the six catenary curves used to compute
the fabric height at a ball's position.

Renders:
  - 4 corner rods (cylinders)
  - 4 perimeter catenaries (W, E, S, N) anchored to rod tops
  - 2 interpolating catenaries (WE at ball.y, SN at ball.x) whose intersection
    gives the surface height where the ball rests
  - 1 ball resting on the surface at that intersection
  - dotted vertical drop lines and small markers on the perimeter catenaries
    showing where the interpolating catenaries are anchored

The fabric surface itself is intentionally NOT drawn — only the curves that
define the height at the ball's position.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics import catenary as cat


# ---- cell parameters ----------------------------------------------------
D_RODS = 0.5
D_FABRIC = 0.6

# Asymmetric rod heights so all six catenaries have visibly different sag.
# Layout (looking down on the cell):
#       NW (0,1) ---- NE (1,1)
#         |              |
#       SW (0,0) ---- SE (1,0)
rod_sw = 0.80
rod_se = 1.00
rod_nw = 0.95
rod_ne = 0.85

# Ball position within the cell (local cell coordinates)
ball_x = 0.32
ball_y = 0.30
BALL_RADIUS = 0.05 * 0.5


# ---- compute the six catenaries -----------------------------------------
# W / E run along y (between south-and-north rods)
cat_w = cat.findcatenaryparameters(D_FABRIC, D_RODS, rod_sw, rod_nw)
cat_e = cat.findcatenaryparameters(D_FABRIC, D_RODS, rod_se, rod_ne)

# S / N run along x (between west-and-east rods)
cat_s = cat.findcatenaryparameters(D_FABRIC, D_RODS, rod_sw, rod_se)
cat_n = cat.findcatenaryparameters(D_FABRIC, D_RODS, rod_nw, rod_ne)

# Heights of the W/E catenaries at the ball's y → endpoints of the WE catenary
height_w_y = cat.catenary(ball_y, cat_w)
height_e_y = cat.catenary(ball_y, cat_e)
cat_we = cat.findcatenaryparameters(D_FABRIC, D_RODS, height_w_y, height_e_y)

# Heights of the S/N catenaries at the ball's x → endpoints of the SN catenary
height_s_x = cat.catenary(ball_x, cat_s)
height_n_x = cat.catenary(ball_x, cat_n)
cat_sn = cat.findcatenaryparameters(D_FABRIC, D_RODS, height_s_x, height_n_x)

# Surface height at the ball
surface_z = cat.catenary(ball_x, cat_we)
ball_center_z = surface_z + BALL_RADIUS


# ---- helpers ------------------------------------------------------------
def sample_curve(params, n=80, lo=0.0, hi=D_RODS):
    t = np.linspace(lo, hi, n)
    z = cat.catenary(t, params)
    return t, z


def draw_rod(ax, x, y, z, radius=0.012, sectors=20, color="0.45", zorder=1):
    theta = np.linspace(0, 2 * np.pi, sectors)
    cx = x + radius * np.cos(theta)
    cy = y + radius * np.sin(theta)
    bottom = np.full_like(cx, 0.0)
    top = np.full_like(cx, z)

    # side wall
    side = []
    for k in range(sectors - 1):
        side.append([
            (cx[k], cy[k], bottom[k]),
            (cx[k + 1], cy[k + 1], bottom[k + 1]),
            (cx[k + 1], cy[k + 1], top[k + 1]),
            (cx[k], cy[k], top[k]),
        ])
    side_coll = Poly3DCollection(side, facecolor=color, edgecolor="none")
    side_coll.set_zorder(zorder)
    ax.add_collection3d(side_coll)

    # top cap
    cap = [list(zip(cx, cy, top))]
    cap_coll = Poly3DCollection(cap, facecolor=color, edgecolor="none")
    cap_coll.set_zorder(zorder)
    ax.add_collection3d(cap_coll)


def draw_sphere(ax, x, y, z, radius, color="0.15", n=24, zorder=20):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    sx = x + radius * np.outer(np.cos(u), np.sin(v))
    sy = y + radius * np.outer(np.sin(u), np.sin(v))
    sz = z + radius * np.outer(np.ones_like(u), np.cos(v))
    surf = ax.plot_surface(sx, sy, sz, color=color, linewidth=0, antialiased=True, shade=True)
    surf.set_zorder(zorder)


# ---- plot ---------------------------------------------------------------
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")
ax.set_proj_type("ortho")
# Disable matplotlib's per-frame z-sorting so we can control draw order
# explicitly via zorder. This lets the catenaries render on top of rods.
ax.computed_zorder = False

# Draw order (via zorder, since computed_zorder=False):
#   1  floor footprint
#   2  rods
#  10  perimeter catenaries
#  11  interpolating catenaries
#  12  anchor markers and rod-top dots
#  13  vertical drop line
#  20  ball

# floor footprint of the cell (just the four edges, very faint)
floor_x = [0, D_RODS, D_RODS, 0, 0]
floor_y = [0, 0, D_RODS, D_RODS, 0]
ax.plot(floor_x, floor_y, [0] * 5, color="0.7", lw=0.6, zorder=1)

# rods
for (rx, ry, rz) in [
    (0.0, 0.0, rod_sw),
    (D_RODS, 0.0, rod_se),
    (0.0, D_RODS, rod_nw),
    (D_RODS, D_RODS, rod_ne),
]:
    draw_rod(ax, rx, ry, rz, zorder=2)

# perimeter catenaries (the four "rod-to-rod" ones)
PERIM_COLOR = "0.6"
PERIM_LW = 2.0

ys, zw = sample_curve(cat_w)
ax.plot(np.zeros_like(ys), ys, zw, color=PERIM_COLOR, lw=PERIM_LW, zorder=10)

ys, ze = sample_curve(cat_e)
ax.plot(np.full_like(ys, D_RODS), ys, ze, color=PERIM_COLOR, lw=PERIM_LW, zorder=10)

xs, zs = sample_curve(cat_s)
ax.plot(xs, np.zeros_like(xs), zs, color=PERIM_COLOR, lw=PERIM_LW, zorder=10)

xs, zn = sample_curve(cat_n)
ax.plot(xs, np.full_like(xs, D_RODS), zn, color=PERIM_COLOR, lw=PERIM_LW, zorder=10)

# interpolating catenaries
INTERP_COLOR = "0.15"
INTERP_LW = 2.4

xs, zwe = sample_curve(cat_we)
ax.plot(xs, np.full_like(xs, ball_y), zwe, color=INTERP_COLOR, lw=INTERP_LW, zorder=11)

ys, zsn = sample_curve(cat_sn)
ax.plot(np.full_like(ys, ball_x), ys, zsn, color=INTERP_COLOR, lw=INTERP_LW, zorder=11)

# rod-top dots and anchor markers for the interpolating catenaries
for (rx, ry, rz) in [
    (0.0, 0.0, rod_sw),
    (D_RODS, 0.0, rod_se),
    (0.0, D_RODS, rod_nw),
    (D_RODS, D_RODS, rod_ne),
]:
    ax.scatter([rx], [ry], [rz], color="0.2", s=18, depthshade=False, zorder=12)

ANCHOR_COLOR = "0.15"
ax.scatter([0.0], [ball_y], [height_w_y], color=ANCHOR_COLOR, s=30, depthshade=False, zorder=12)
ax.scatter([D_RODS], [ball_y], [height_e_y], color=ANCHOR_COLOR, s=30, depthshade=False, zorder=12)
ax.scatter([ball_x], [0.0], [height_s_x], color=ANCHOR_COLOR, s=30, depthshade=False, zorder=12)
ax.scatter([ball_x], [D_RODS], [height_n_x], color=ANCHOR_COLOR, s=30, depthshade=False, zorder=12)

# vertical drop from ball to ground
ax.plot([ball_x, ball_x], [ball_y, ball_y], [0.0, surface_z],
        color="0.55", lw=0.8, linestyle=":", zorder=13)

# ball
draw_sphere(ax, ball_x, ball_y, ball_center_z, BALL_RADIUS, zorder=20)

# axes / framing — show rods all the way down to z=0 with a tight x-y margin.
margin = 0.04
ax.set_xlim(-margin, D_RODS + margin)
ax.set_ylim(-margin, D_RODS + margin)
z_lo = 0.0
z_hi = max(rod_sw, rod_se, rod_nw, rod_ne) + 0.02
ax.set_zlim(z_lo, z_hi)
ax.set_box_aspect((D_RODS + 2 * margin, D_RODS + 2 * margin, (z_hi - z_lo) * 1.0))

ax.view_init(elev=57, azim=-38)

ax.set_axis_off()

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
out_path = Path(__file__).parent / "cell_catenaries.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {out_path}")
