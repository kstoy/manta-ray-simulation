from src.constants import NE, NW, SW, SE
from src.controllers.controller_base import Controller


class MassSortController(Controller):
    """Controller that sorts balls by mass using sensor feedback."""

    def update(self, i, j, timestep, sensors):
        if j == 0:
            if sensors[NE] < 0.0001:
                return 0.5
            else:
                return 1.5
        elif j == 1:
            if sensors[SE] < 0.0001:
                return 0.5
            else:
                return 1.5

            if j == 1:
                if sensors[SW] < self.config.TARGET_WEIGHT:
                    return 0.5
                else:
                    return 1.5
        elif i > 1:
            if j == 0:
                if sensors[NW] > 0:
                    return 1.5
                else:
                    return 0.5
            if j == 1:
                if sensors[SW] > 0:
                    return 1.5
                else:
                    return 0.5
        # Default return for unhandled cases
        return 1.0
