"""Save and load simulation data files."""
import pickle
from pathlib import Path

DEFAULT_PATH = "output/simdata.pkl"


def save(path, rodsstates, ballsstates, ballsradiuses, config):
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "rodsstates": rodsstates,
            "ballsstates": ballsstates,
            "ballsradiuses": ballsradiuses,
            "config": config,
        }, f)
    print(f"Simulation data saved to {path}")


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
