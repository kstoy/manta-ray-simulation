"""Controller registry for surface control strategies."""

from src.controllers.controller_base import Controller
from src.controllers.squarecontroller import SquareController
from src.controllers.squarecontroller_nonedeterministic_push import SquareControllerPush
from src.controllers.squarecontroller_nonedeterministic_pull import SquareControllerPull
from src.controllers.mass_sort_controller import MassSortController


# Controller registry - maps string names to controller classes
CONTROLLER_REGISTRY = {
    "square": SquareController,
    "square_push": SquareControllerPush,
    "square_pull": SquareControllerPull,
    "mass_sort": MassSortController,
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
