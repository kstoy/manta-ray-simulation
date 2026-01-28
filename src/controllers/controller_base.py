from abc import ABC, abstractmethod


class Controller(ABC):
    """Base class for all surface controllers."""

    def __init__(self, config):
        """
        Initialize the controller.

        Args:
            config: SimConfig instance with simulation parameters
        """
        self.config = config

    @abstractmethod
    def update(self, i: int, j: int, timestep: int, sensors) -> float:
        """
        Compute desired rod height for position (i, j).

        Args:
            i: Rod x-index
            j: Rod y-index
            timestep: Current simulation timestep
            sensors: Sensor readings at this rod [NE, NW, SW, SE]

        Returns:
            Desired rod height (typically 0.5 to 1.5)
        """
        pass
