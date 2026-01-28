import numpy as np
from src.controllers.controller_base import Controller


class SquareControllerPull(Controller):
    """Controller using a pull pattern - lowers rod when sensor detects weight."""

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

        NE_SW = np.array([1, 0, 1, 0])
        NW_SE = np.array([0, 1, 0, 1])

        N_SW = np.array([1, 1, 1, 0])
        N_SE = np.array([1, 1, 0, 1])
        S_NE = np.array([1, 0, 1, 1])
        S_NW = np.array([0, 1, 1, 1])

        A = np.array([1, 1, 1, 1])

        self.pattern = np.flip(np.array([
            [I, I, I, I, I, N_SE, E, E, E, I],
            [N, N, N, N, N, N_SE, E, E, E, I],
            [N, N, N, N, N, N_SE, E, E, E, I],
            [N, N, N, N, N, N_SE, E, E, E, I],
            [N, N_SW, N_SW, NE_SW, A, A, E, E, E, I],
            [I, W, W, W, A, A, S_NE, S_NE, S_NE, S],
            [I, W, W, W, S_NW, S, S, S, S, S],
            [I, W, W, W, S_NW, S, S, S, S, S],
            [I, W, W, W, S_NW, S, S, S, S, S],
            [I, W, W, W, W, I, I, I, I, I],
        ]), 0)

    def update(self, i, j, timestep, sensors):
        # Inverted logic: lower rod when sensor detects weight
        if np.logical_and(self.pattern[j][i], sensors).any():
            return 0.5
        else:
            return 1.5
