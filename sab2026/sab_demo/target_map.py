"""Target (final) cells for the sab_demo convergence analysis.

Convergence requires that every cell marked 'T' below contains at least one
ball (ball center inside the cell). Edit the map to choose which cells are
targets: 'T' = target, '.' = ignored.

Layout mirrors DIRECTION_MAP in sab_demo.py exactly:
  - Written top-to-bottom = visual row 7 (top) down to row 1 (bottom).
  - np.flip(..., 0) makes index [0] the BOTTOM row, so TARGET_MAP is indexed
    TARGET_MAP[cell_row, cell_col] with cell_row 0 = bottom, matching the
    controller's priority_map[cy, cx] convention.
  - 15 columns (cell_col 0..14), 7 rows (cell_row 0..6).

Pre-filled with the 17 'I' (idle) cells of the current DIRECTION_MAP as a
starting point.
"""
import numpy as np

TARGET_MARKER = "T"

#                col: 0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
TARGET_MAP = np.flip(np.array([
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # row 7 (top)
    ['.', '.', 'T', 'T', 'T', '.', 'T', 'T', 'T', '.', 'T', 'T', '.', '.', '.'],  # row 6
    ['.', '.', 'T', '.', '.', '.', 'T', '.', 'T', '.', 'T', '.', 'T', '.', '.'],  # row 5
    ['.', '.', 'T', 'T', 'T', '.', 'T', 'T', 'T', '.', 'T', 'T', 'T', '.', '.'],  # row 4
    ['.', '.', '.', '.', 'T', '.', 'T', '.', 'T', '.', 'T', '.', 'T', '.', '.'],  # row 3
    ['.', '.', 'T', 'T', 'T', '.', 'T', '.', 'T', '.', 'T', 'T', 'T', '.', '.'],  # row 2
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  # row 1 (bottom)
]), 0)


def target_mask():
    """Boolean array (n_cell_rows, n_cell_cols) indexed [cell_row, cell_col];
    True where the cell is a target. Matches priority_map[cy, cx] indexing."""
    return TARGET_MAP == TARGET_MARKER


def target_coords():
    """List of (cell_col, cell_row) index pairs for target cells (col=cx, row=cy)."""
    ys, xs = np.where(target_mask())
    return list(zip(xs.tolist(), ys.tolist()))


if __name__ == "__main__":
    mask = target_mask()
    print(f"{int(mask.sum())} target cells")
    print("Visual layout (top row = cell_row 6):")
    for cy in range(mask.shape[0] - 1, -1, -1):
        print(f"row {cy}: " + " ".join("T" if mask[cy, cx] else "." for cx in range(mask.shape[1])))
