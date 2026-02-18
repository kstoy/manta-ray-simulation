# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics-based surface simulation that models balls rolling on a fabric surface suspended by a grid of controllable rods. Uses catenary curves to model fabric sag between attachment points and XPBD (Extended Position Based Dynamics) for collision and friction physics.

**Purpose**: Explore surface control strategies to direct ball movement across a deformable fabric surface by raising/lowering rod heights.

## Running the Code

```bash
python run.py                                    # Run simulation with default experiment
python run.py --experiment experiments/itu_demo.py   # Use a custom experiment
python run.py --output mydata.pkl                # Custom data file path

python visualize.py opengl                       # Interactive 3D viewer
python visualize.py video                        # Export MP4
python visualize.py matplotlib                   # Matplotlib animation
python visualize.py opengl --input mydata.pkl    # Load from custom path

python scripts/performancetest.py               # Benchmark vs ball count
python scripts/profilesimulation.py             # Profile performance bottlenecks
```

## Architecture

### Data Flow
```
Sensors → Controller → RodsState.update() → Catenary Surface → XPBD Physics → BallsState
```

### Main Simulation Loop ([src/simulation.py](src/simulation.py))
1. **Sensor aggregation**: Each ball's mass contributes to 4 surrounding rod corners (NE, NW, SW, SE)
2. **Controller update**: `controller.update_all(timestep, sensors)` returns desired heights for all rods
3. **Rod P-control**: `rod_z += K * (desired - current)` smoothly adjusts rod heights
4. **Physics substeps**: XPBD solver handles surface contact, ball-ball collisions, friction with spin

### Core Components
- **RodsState** ([src/state/rods.py](src/state/rods.py)): Rod grid positions, sensors array, controller integration
- **BallsState** ([src/state/balls.py](src/state/balls.py)): Ball positions `r`, velocities `v`, angular velocities `w`, masses `m`, radii `R`
- **simcorexpbd** ([src/physics/simcorexpbd.py](src/physics/simcorexpbd.py)): XPBD physics; surface height via `rodsstate.surfacejet(x, y)` → `(z, dz/dx, dz/dy)`

### Sensor System
Sensors are a `(GRIDSIZEX, GRIDSIZEY, 4)` array where the 4 channels represent directional weight from balls in each quadrant (`NE=0, NW=1, SW=2, SE=3`, defined in [src/config.py](src/config.py)).

## Key Parameters ([src/config.py](src/config.py))

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRIDSIZEX` / `GRIDSIZEY` | 10 / 10 | Rod grid dimensions (auto-derived from `DIRECTION_MAP` if present) |
| `D` | 1.0 | Rod spacing (meters) |
| `LF` | 1.45 | Fabric length factor (controls sag amount) |
| `DT` | 0.1 | Physics timestep |
| `K` | 0.2 | Rod height P-control gain |
| `NBALL` | 20 | Number of balls |
| `BALL_INIT` | `"outside_rectangle"` | Ball init strategy: `"grid_uniform"`, `"random"`, `"center_cluster"`, `"outside_rectangle"` |
| `CONTROLLER` | `"blocking"` | Controller: `"blocking"`, `"nonblocking"`, `"priority"` |

## Experiment Files ([experiments/](experiments/))

Experiments are plain Python files with variable assignments. `load_experiment()` in `run.py` reads them and builds a `SimConfig`.

```python
# experiments/my_experiment.py
import numpy as np

NBALL = 20
CONTROLLER    = "nonblocking"   # or "blocking" / "priority"
BALL_INIT     = "outside_rectangle"
RESPAWN       = True
MAXSIMULATIONSTEPS = 2750

DIRECTION_MAP = np.flip(np.array([
    ['S', 'S', ...],  # top row (visually)
    ...
    ['N', 'N', ...],  # bottom row (visually)
]), 0)
# GRIDSIZEX and GRIDSIZEY are derived automatically from DIRECTION_MAP.shape
```

Direction values: `N`, `S`, `E`, `W`, `I` (idle). Priority controller also accepts multi-char strings like `"NE"` (try N first, fall back to E).

## Creating a New Controller

1. Create class in `src/controllers/` extending `Controller` from [controller_base.py](src/controllers/controller_base.py)
2. Set `self.direction_map` before calling `super().__init__(config)` if using the shared quadrant mapping
3. Implement `update(i, j, timestep, sensors) -> float` returning desired rod height (0.5 = lower, 1.5 = raise)
4. **Optional**: Implement `update_all(timestep, sensors) -> ndarray` for vectorized performance
5. Register in [src/controllers/\_\_init\_\_.py](src/controllers/__init__.py) `CONTROLLER_REGISTRY`

## Dependencies

numpy, scipy, pygltflib, PyOpenGL, matplotlib, ffmpeg (for video export)
