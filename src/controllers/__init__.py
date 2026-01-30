"""Controller registry for surface control strategies."""

from src.controllers.controller_base import Controller
from src.controllers.squarecontroller_nonedeterministic_push import SquareControllerPush
from src.controllers.squarecontroller_nonedeterministic_pull import SquareControllerPull

try:
    from src.controllers.weightsortcontroller import (
        WeightSortController,
        WeightSortRadialController,
        WeightSortGradientController,
    )
    _has_weightsort = True
except ImportError:
    _has_weightsort = False

try:
    from src.controllers.testslopecontroller import TestSlopeController
    _has_testslope = True
except ImportError:
    _has_testslope = False


# Controller registry - maps string names to controller classes
CONTROLLER_REGISTRY = {
    "square_push": SquareControllerPush,
    "square_pull": SquareControllerPull,
}

if _has_weightsort:
    CONTROLLER_REGISTRY["weight_sort"] = WeightSortController
    CONTROLLER_REGISTRY["weight_sort_radial"] = WeightSortRadialController
    CONTROLLER_REGISTRY["weight_sort_gradient"] = WeightSortGradientController

if _has_testslope:
    CONTROLLER_REGISTRY["test_slope"] = TestSlopeController


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
