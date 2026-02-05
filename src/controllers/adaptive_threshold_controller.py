import numpy as np
from src.controllers.controller_base import Controller
from src.constants import NE, NW, SW, SE


class AdaptiveThresholdController(Controller):

    def __init__(self, config, alpha=0.1, weight_threshold_init=8.37):
        super().__init__(config)
        self.alpha = alpha
        self.UP = 1.5
        self.DOWN = 0.5

        # Per-rod weight threshold: objects below go south, above go east
        self.weight_threshold = np.full((config.GRIDSIZEX, config.GRIDSIZEY), weight_threshold_init)

        # Per-rod constant desired fraction of objects that should go east
        # At row i, fraction_east = (n-i-1)/(n-i) so each row passes 1/n south
        n = config.GRIDSIZEX
        self.target_fraction_east = np.zeros((config.GRIDSIZEX, config.GRIDSIZEY))
        for i in range(n):
            remaining = n - i
            self.target_fraction_east[i, :] = (remaining - 1) / remaining

        # Per-rod moving averages of eastward and southward movements
        self.avg_east = np.zeros((config.GRIDSIZEX, config.GRIDSIZEY))
        self.avg_south = np.zeros((config.GRIDSIZEX, config.GRIDSIZEY))

        # Previous sensor values per rod for detecting departures
        self._prev = np.zeros((config.GRIDSIZEX, config.GRIDSIZEY, 4))

    def update(self, i, j, timestep, sensors):
        prev = self._prev[i, j]

        # Eastward: NW→NE or SW→SE
        moved_east = ((prev[NW] > 0 and sensors[NW] == 0 and sensors[NE] > 0) or
                      (prev[SW] > 0 and sensors[SW] == 0 and sensors[SE] > 0))

        # Southward: NW→SW or NE→SE
        moved_south = ((prev[NW] > 0 and sensors[NW] == 0 and sensors[SW] > 0) or
                       (prev[NE] > 0 and sensors[NE] == 0 and sensors[SE] > 0))

        # Update moving averages
        self.avg_east[i, j] = self.avg_east[i, j] * (1 - self.alpha) + self.alpha * float(moved_east)
        self.avg_south[i, j] = self.avg_south[i, j] * (1 - self.alpha) + self.alpha * float(moved_south)

        # Adjust weight threshold if actual fraction doesn't match target
        total = self.avg_east[i, j] + self.avg_south[i, j]
        if total > 0:
            actual_fraction_east = self.avg_east[i, j] / total
            error = actual_fraction_east - self.target_fraction_east[i, j]
            # Too many going east → raise threshold; too few → lower threshold
            self.weight_threshold[i, j] += self.alpha * error

        self._prev[i, j] = sensors

        # Control: default is high
        desired = self.UP
        threshold = self.weight_threshold[i, j]

        # Object to the N with weight below threshold → lower (push south)
        if (sensors[NW] > 0 and sensors[NW] < threshold) or (sensors[NE] > 0 and sensors[NE] < threshold):
            desired = self.DOWN

        # Object to the W with weight above threshold → lower (push east)
        if (sensors[NW] > 0 and sensors[NW] > threshold) or (sensors[SW] > 0 and sensors[SW] > threshold):
            desired = self.DOWN

        return desired
