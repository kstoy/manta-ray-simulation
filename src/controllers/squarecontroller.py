import numpy as np
from src.controllers.controller_base import Controller


class SquareController(Controller):
    """Controller using a predefined grid pattern for ball direction."""

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
            [S,  S,  S,  S,  S, SW, SW, SW, SW, SW],
            [SE, SE, SE, SE, SE, SW, S_NW, S_NW, S_NW, W],
            [S_NE, S_NE, S_NE, S_NE, SE, SW, SW, SW, SW, SW],
            [SE, SE, SE, SE, SE, SW, W, W, W, W],
            [E, E, E, E, I, I, W, W, W, W],
            [E, E, E, E, I, I, W, W, W, W],
            [NE, NE, NE, NE, NE, NW, W, W, W, W],
            [N_SE, N_SE, N_SE, N_SE, NE, NW, NW, NW, NW, NW],
            [NE, NE, NE, NE, NE, NW, N_SW, N_SW, N_SW, W],
            [N, N, N, N, NE, NW, NW, NW, NW, NW]
        ]), 0)

    def update(self, i, j, timestep, sensors):
        if np.logical_and(self.pattern[j][i], sensors).any():
            return 1.5
        else:
            return 0.5
