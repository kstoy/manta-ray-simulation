"""Controller registry for surface control strategies."""

from src.controllers.controller_base import Controller
from src.controllers.controller_blocking import ControllerBlocking
from src.controllers.controller_blocking_weighted import ControllerBlockingWeighted
from src.controllers.controller_distributed_coverage import ControllerDistributedCoverage
from src.controllers.controller_greedy_bin import ControllerGreedyBin
from src.controllers.controller_greedy_bin_phased import ControllerGreedyBinPhased
from src.controllers.controller_nonblocking import ControllerNonBlocking
from src.controllers.controller_priority import ControllerPriority
from src.controllers.controller_priority_delayed import ControllerPriorityDelayed
from src.controllers.controller_priority_preemptive import ControllerPriorityPreemptive
from src.controllers.controller_random_pair import ControllerRandomPair
from src.controllers.controller_stochastic_assembly import ControllerStochasticAssembly

CONTROLLER_REGISTRY = {
    "blocking":            ControllerBlocking,
    "blocking_weighted":   ControllerBlockingWeighted,
    "distributed_coverage": ControllerDistributedCoverage,
    "greedy_bin":          ControllerGreedyBin,
    "greedy_bin_phased":   ControllerGreedyBinPhased,
    "nonblocking":         ControllerNonBlocking,
    "priority":            ControllerPriority,
    "priority_delayed":    ControllerPriorityDelayed,
    "priority_preemptive": ControllerPriorityPreemptive,
    "random_pair":         ControllerRandomPair,
    "stochastic_assembly": ControllerStochasticAssembly,
}


def get_controller(name_or_factory, config):
    """
    Get a controller instance by name or factory callable.

    Args:
        name_or_factory: Controller name string, or a callable(config) -> Controller
        config: SimConfig instance
    """
    if callable(name_or_factory):
        return name_or_factory(config)

    if name_or_factory not in CONTROLLER_REGISTRY:
        available = ", ".join(CONTROLLER_REGISTRY.keys())
        raise ValueError(f"Unknown controller '{name_or_factory}'. Available: {available}")

    return CONTROLLER_REGISTRY[name_or_factory](config)


__all__ = ["Controller", "get_controller", "CONTROLLER_REGISTRY"]
