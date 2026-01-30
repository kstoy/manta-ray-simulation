# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics-based surface simulation that models balls rolling on a fabric surface suspended by a grid of controllable rods. Uses catenary curves to model fabric sag between attachment points and XPBD (Extended Position Based Dynamics) for collision and friction physics.

**Purpose**: Explore surface control strategies to direct ball movement across a deformable fabric surface by raising/lowering rod heights.

## Running the Code

```bash
python run_simulation.py              # Run simulation, outputs GLTF files to output/
python scripts/performancetest.py     # Benchmark vs ball count
python scripts/profilesimulation.py   # Profile performance bottlenecks
```

## Architecture

### Data Flow
```
Sensors → Controller → RodsState.update() → Catenary Surface → XPBD Physics → BallsState
```

### Main Simulation Loop ([src/simulation.py](src/simulation.py))
1. **Sensor aggregation**: Each ball's mass contributes to 4 surrounding rod corners (NE, NW, SW, SE directions)
2. **Controller update**: For each rod, `controller.update(i, j, timestep, sensors)` returns desired height
3. **Rod P-control**: `rod_z += K * (desired - current)` smoothly adjusts rod heights
4. **Physics substeps**: XPBD solver handles surface contact, ball-ball collisions, friction with spin

### Core Components
- **RodsState** ([src/rodstate.py](src/rodstate.py)): Manages rod grid positions, sensors array, calls controller. Change `self.controller` to use different control strategy.
- **BallsState** ([src/ballstate.py](src/ballstate.py)): Ball positions `r`, velocities `v`, angular velocities `w`, masses `m`, radii `R`
- **simcorexpbd** ([src/physics/simcorexpbd.py](src/physics/simcorexpbd.py)): XPBD physics with surface height lookup via `rodsstate.surfacejet(x, y)` returning `(z, dz/dx, dz/dy)`

### Sensor System
Sensors are a `(GRIDSIZEX, GRIDSIZEY, 4)` array where the 4 channels represent directional weight from balls in each quadrant (NE=0, NW=1, SW=2, SE=3 per [src/constants.py](src/constants.py)).

## Key Parameters ([src/config.py](src/config.py))

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRIDSIZEX` / `GRIDSIZEY` | 10 / 10 | Rod grid dimensions |
| `D` | 1.0 | Rod spacing (meters) |
| `LF` | 1.45 | Fabric length factor (controls sag amount) |
| `DT` | 0.1 | Physics timestep |
| `K` | 0.2 | Rod height P-control gain |
| `CONTROLLER` | "square_push" | Controller type: "square_push", "square_pull", "weight_sort", "weight_sort_radial", "weight_sort_gradient", "test_slope" |
| `NBALL` | (derived) | Number of balls = `(GRIDSIZEX-1) * (GRIDSIZEY-1)` |

## Creating a New Controller

1. Create class in `src/controllers/` extending `Controller` from [controller_base.py](src/controllers/controller_base.py)
2. Implement `update(i, j, timestep, sensors) -> float` returning desired rod height (typically 0.5 to 1.5)
3. **Optional**: Implement `update_all(timestep, sensors) -> ndarray` for vectorized performance (10-20x faster)
4. Register controller in [src/controllers/\_\_init\_\_.py](src/controllers/__init__.py) `CONTROLLER_REGISTRY`
5. Switch controllers via config: `SimConfig(CONTROLLER="your_controller_name")`

Example: [squarecontroller_nonedeterministic_push.py](src/controllers/squarecontroller_nonedeterministic_push.py) uses pattern masks to create directional movement zones.

## Dependencies

numpy, scipy, pygltflib
