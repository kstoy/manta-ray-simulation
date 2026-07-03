#!/usr/bin/env python
"""Run the surface simulation and save results to a data file.

Usage:
    python run.py                                        # simulate and save data
    python run.py --no-save                              # simulate only, no data file
    python run.py --experiment experiments/itu_demo_stable.py   # use a custom experiment
    python run.py --output mydata.pkl                    # custom output path
"""
import argparse
import dataclasses
import importlib.util
import time

from src.simulation import simulation
from src.config import SimConfig
from src import simdata

DEFAULT_EXPERIMENT = "experiments/center_demo.py"


def load_experiment(path):
    spec = importlib.util.spec_from_file_location("_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    field_names = {f.name for f in dataclasses.fields(SimConfig)}
    kwargs = {k: getattr(module, k) for k in field_names if hasattr(module, k)}
    config = SimConfig(**kwargs)

    if hasattr(module, "DIRECTION_MAP"):
        dm = module.DIRECTION_MAP
        # Derive grid size from map shape if not set explicitly in experiment
        if not hasattr(module, "GRIDSIZEX"):
            config.GRIDSIZEX = dm.shape[1] + 1
        if not hasattr(module, "GRIDSIZEY"):
            config.GRIDSIZEY = dm.shape[0] + 1
        if isinstance(config.CONTROLLER, str):
            from src.controllers import CONTROLLER_REGISTRY
            name = config.CONTROLLER
            sm = getattr(module, "SORTER_MAP", None)
            if sm is not None:
                config.CONTROLLER = lambda cfg, n=name, d=dm, s=sm: CONTROLLER_REGISTRY[n](cfg, d, s)
            else:
                config.CONTROLLER = lambda cfg, n=name, d=dm: CONTROLLER_REGISTRY[n](cfg, d)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manta ray surface simulation")
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT,
        help=f"Experiment file (default: {DEFAULT_EXPERIMENT})"
    )
    parser.add_argument(
        "--output", default=simdata.DEFAULT_PATH,
        help=f"Data file path (default: {simdata.DEFAULT_PATH})"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Run simulation without saving data to file"
    )
    args = parser.parse_args()

    config = load_experiment(args.experiment)
    print(f"Experiment: {args.experiment}")
    print("Running simulation...", end="", flush=True)
    start = time.time()
    save = not args.no_save
    rodsstates, ballsstates, ballsradiuses, channels = simulation(config=config, visualization=save)
    print(f" done ({time.time() - start:.1f}s)")

    if save:
        simdata.save(args.output, rodsstates, ballsstates, ballsradiuses, config, channels=channels)
