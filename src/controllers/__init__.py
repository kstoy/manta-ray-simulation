"""Controller registry for surface control strategies."""

from src.controllers.controller_base import Controller
from src.controllers.squarecontroller_nonedeterministic_push import SquareControllerPush
from src.controllers.squarecontroller_nonedeterministic_pull import SquareControllerPull
from src.controllers.weightsortcontroller import (
    WeightSortController,
    WeightSortRadialController,
    WeightSortGradientController,
)
from src.controllers.testslopecontroller import TestSlopeController


# Controller registry - maps string names to controller classes
CONTROLLER_REGISTRY = {
    "square_push": SquareControllerPush,
    "square_pull": SquareControllerPull,
    "weight_sort": WeightSortController,
    "weight_sort_radial": WeightSortRadialController,
    "weight_sort_gradient": WeightSortGradientController,
    "test_slope": TestSlopeController,
}


def get_controller(name: str, config):
    """
    Get a controller instance by name.

    Args:
        name: Controller name (e.g., "square_push", "square_pull")
        config: SimConfig instance

    Returns:
        Controller instance

    Raises:
        ValueError: If controller name is not registered
    """
    if name not in CONTROLLER_REGISTRY:
        available = ", ".join(CONTROLLER_REGISTRY.keys())
        raise ValueError(f"Unknown controller '{name}'. Available: {available}")

    controller_class = CONTROLLER_REGISTRY[name]
    return controller_class(config)


__all__ = ["Controller", "get_controller", "CONTROLLER_REGISTRY"]
