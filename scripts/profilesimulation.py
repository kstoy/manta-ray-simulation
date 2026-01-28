import cProfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import simulation
from src.config import SimConfig

config = SimConfig()

cProfile.runctx(
    'simulation.simulation(config=config, visualization=False)',
    globals={'simulation': simulation, 'config': config},
    locals={},
    sort='cumtime'
)
