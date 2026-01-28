import numpy as np
from src.controllers.controller_base import Controller


class SquareControllerPush(Controller):
    """Controller using a push pattern - raises rod when sensor detects weight."""

    def __init__(self, config):
        super().__init__(config)

        # Direction masks [NE, NW, SW, SE]
        I = np.array([0, 0, 0, 0])

        NE = np.array([1, 0, 0, 0])
        NW = np.array([0, 1, 0, 0])
        SW = np.array([0, 0, 1, 0])
        SE = np.array([0, 0, 0, 1])

        E = np.array([1, 0, 0, 1])
        N = np.array([1, 1, 0, 0])
        W = np.array([0, 1, 1, 0])
        S = np.array([0, 0, 1, 1])

        N_SW = np.array([1, 1, 1, 0])
        N_SE = np.array([1, 1, 0, 1])
        S_NE = np.array([1, 0, 1, 1])
        S_NW = np.array([0, 1, 1, 1])

        A = np.array([1, 1, 1, 1])

        self.pattern = np.flip(np.array([
            [SE, S, S, S, S, SW, SW, SW, SW, SW],
            [SE, S, S, S, S, SW, W, W, W, W],
            [SE, S, S, S, S, SW, W, W, W, W],
            [SE, S, S, S, S, SW, W, W, W, W],
            [SE, SE, SE, SE, I, I, W, W, W, W],
            [E, E, E, E, I, I, NW, NW, NW, NW],
            [E, E, E, E, NE, N, N, N, N, NW],
            [E, E, E, E, NE, N, N, N, N, NW],
            [E, E, E, E, NE, N, N, N, N, NW],
            [NE, NE, NE, NE, NE, N, N, N, N, NW],
        ]), 0)

    def update(self, i, j, timestep, sensors):
        if np.logical_and(self.pattern[j][i], sensors).any():
            return 1.5
        else:
            return 0.5

    def update_all(self, timestep: int, sensors: np.ndarray) -> np.ndarray:
        """
        Vectorized update for all rods at once.

        Args:
            timestep: Current simulation timestep
            sensors: Sensor array of shape (GRIDSIZEX, GRIDSIZEY, 4)

        Returns:
            Array of desired heights with shape (GRIDSIZEX, GRIDSIZEY)
        """
        # Pattern is indexed as [j][i] (y, x), sensors as [i][j] (x, y)
        # Transpose pattern to align: (GRIDSIZEY, GRIDSIZEX, 4) -> (GRIDSIZEX, GRIDSIZEY, 4)
        pattern_aligned = self.pattern.transpose(1, 0, 2)

        # Element-wise logical AND between pattern and sensors, then check if any direction matches
        # Shape: (GRIDSIZEX, GRIDSIZEY, 4) -> (GRIDSIZEX, GRIDSIZEY)
        matches = np.logical_and(pattern_aligned, sensors).any(axis=2)

        # Return 1.5 where pattern matches sensor, 0.5 otherwise
        return np.where(matches, 1.5, 0.5)
