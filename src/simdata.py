"""Save and load simulation data files."""
import pickle
from pathlib import Path

DEFAULT_PATH = "output/simdata.pkl"


def save(path, rodsstates, ballsstates, ballsradiuses, config, channels=None):
    import dataclasses
    # CONTROLLER may be a lambda that can't be pickled; replace with None for storage
    saveable_config = dataclasses.replace(config, CONTROLLER=None)
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "rodsstates": rodsstates,
            "ballsstates": ballsstates,
            "ballsradiuses": ballsradiuses,
            "config": saveable_config,
            "channels": channels,
        }, f)
    print(f"Simulation data saved to {path}")


def load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Backwards-compatible default for pickles saved before channels existed.
    data.setdefault("channels", None)
    return data
